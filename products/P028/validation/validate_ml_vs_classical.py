"""Validation 5: the learned ranker against the classical Pyramid rule.

This is the AI benchmark. The classical rules were implemented, tested and
validated first (validation 3 and 4); the ranker is trained on the candidates
their own geometric search produces, and both are scored on the same frames
with the same candidate lists, so the only thing that differs is the decision.

What is measured, in order:

* 5a dataset: size, class balance, and the search ceiling, which no decision
  rule can exceed;
* 5b training cost;
* 5c row-level ranking quality on held-out frames, and the ablation that says
  how much of it comes from the photometric features the simulator flatters;
* 5d calibration of the confidence output: reliability table, Brier score,
  expected calibration error;
* 5e frame-level identification and false-identification rates against the
  Pyramid rule, over operating points the ranker was and was not trained on;
* 5f the threshold curve -- the operating points the classical rule does not
  have;
* 5g runtime, including the cost of the full scan the ranker needs and the
  early exit the classical rule gets;
* 5h where the ranker fails.

Run: ``python validation/validate_ml_vs_classical.py``   (about 150 s on 2 cores)
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import (
    CAMERA_FOV_DEG,
    CAMERA_PIXELS,
    RATE_HEADER,
    SEED,
    banner,
    finish,
    rate_row,
    report,
    verdict,
)

from skymatch.benchmark import run_trials
from skymatch.camera import CameraModel
from skymatch.dataset import DEFAULT_GRID, build_catalogue_tables, generate_candidate_dataset
from skymatch.identify import SearchConfig, gather_candidates
from skymatch.ranker import (
    LearnedRanker,
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from skymatch.scene import SceneConfig, simulate_scene
from skymatch.triangle import separation_tolerance

TRAIN_FRAMES = 600
TEST_FRAMES = 300
TRAIN_SEED = 1234
TEST_SEED = 5678
BENCH_TRIALS = 150
THRESHOLDS = (0.10, 0.30, 0.50, 0.70, 0.90, 0.99)


def main() -> int:
    passed: list[bool] = []
    cam = CameraModel(fov_deg=CAMERA_FOV_DEG, pixels=CAMERA_PIXELS)

    banner("VALIDATION 5a: the candidate dataset")
    t0 = time.perf_counter()
    tables = build_catalogue_tables(
        tuple(p.magnitude_limit for p in DEFAULT_GRID), cam, SEED
    )
    t_tables = time.perf_counter() - t0
    print(f"    catalogues and pair tables for magnitude limits "
          f"{sorted(tables)}: {t_tables:.1f} s")
    for limit in sorted(tables):
        cat, table = tables[limit]
        print(f"      mag {limit:.1f}: {cat.n_stars:6d} stars, {table.n_pairs:8d} pairs, "
              f"{table.nbytes / 1e6:6.1f} MB")

    t0 = time.perf_counter()
    train = generate_candidate_dataset(TRAIN_FRAMES, TRAIN_SEED, camera=cam, tables=tables)
    t_train_data = time.perf_counter() - t0
    t0 = time.perf_counter()
    test = generate_candidate_dataset(TEST_FRAMES, TEST_SEED, camera=cam, tables=tables)
    t_test_data = time.perf_counter() - t0
    print()
    print(f"{'split':>8} {'frames':>8} {'rows':>8} {'rows/frame':>11} {'positive':>10} "
          f"{'solvable frames':>16} {'seconds':>8}")
    for name, data, secs in (("train", train, t_train_data), ("test", test, t_test_data)):
        print(f"{name:>8} {data.n_frames:8d} {data.n_rows:8d} {data.n_rows / data.n_frames:11.2f} "
              f"{data.positive_fraction:10.4f} "
              f"{data.metadata['solvable_frame_fraction']:16.4f} {secs:8.1f}")
    print("    The split is by frame and by seed: the two sets share the catalogues, which")
    print("    is the split a star tracker faces (the catalogue is fixed, the sky is not).")
    print("    'solvable frames' is the fraction whose candidate list contains the truth;")
    print("    it is the ceiling on every decision rule and it is below 1 because the")
    print("    training grid deliberately includes frames nothing can solve.")
    passed.append(verdict("test rows", float(test.n_rows), 3000.0, mode=">="))
    passed.append(
        verdict("test positive fraction is a minority class", test.positive_fraction, 0.5)
    )

    print()
    banner("VALIDATION 5b: training")
    t0 = time.perf_counter()
    ranker = LearnedRanker(random_state=0).fit(train.features, train.labels)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ablated = LearnedRanker(use_magnitude_features=False, random_state=0).fit(
        train.features, train.labels
    )
    t_fit_ablated = time.perf_counter() - t0
    print(f"    HistGradientBoostingClassifier, 13 features, {train.n_rows} rows")
    print(f"    full model      {t_fit:6.1f} s")
    print(f"    no-photometry   {t_fit_ablated:6.1f} s   ({len(ablated.feature_names)} features)")
    print(f"    dataset + fit   {t_tables + t_train_data + t_fit:6.1f} s total training cost")
    passed.append(verdict("total training cost [s]", t_tables + t_train_data + t_fit, 180.0))

    print()
    banner("VALIDATION 5c: row-level ranking on held-out frames")
    scores = ranker.score(test.features)
    scores_ab = ablated.score(test.features)
    base_rate = test.positive_fraction
    print(f"{'model':>18} {'ROC AUC':>10} {'average precision':>19} {'base rate':>11}")
    for name, sc in (("full", scores), ("no photometry", scores_ab)):
        print(f"{name:>18} {roc_auc_score(test.labels, sc):10.5f} "
              f"{average_precision_score(test.labels, sc):19.5f} {base_rate:11.5f}")
    auc = float(roc_auc_score(test.labels, scores))
    passed.append(verdict("ROC AUC, full model", auc, 0.95, mode=">="))
    print()
    print("    permutation importance (drop in average precision when a column is shuffled),")
    print("    computed on the held-out set")
    rng = np.random.default_rng(SEED)
    importance = ranker.permutation_importance(test.features, test.labels, rng, n_repeats=3)
    order = np.argsort(importance)[::-1]
    for slot in order:
        print(f"      {ranker.feature_names[slot]:<28s} {importance[slot]:10.5f}")
    print("    The photometric columns are flattered by the simulator, which gives the")
    print("    instrument the catalogue's own magnitude scale plus Gaussian noise: no")
    print("    colour term, no zero-point drift, no saturation. The ablated model above is")
    print("    the honest lower bound on this feature set.")

    print()
    banner("VALIDATION 5d: is the confidence a probability?")
    print("    reliability table on held-out candidates, 10 equal-width bins")
    mean_p, freq, counts = reliability_table(scores, test.labels, n_bins=10)
    print(f"{'mean predicted':>16} {'observed':>10} {'count':>8} {'gap':>10}")
    for mp, fr, ct in zip(mean_p, freq, counts, strict=True):
        print(f"{mp:16.4f} {fr:10.4f} {ct:8d} {mp - fr:10.4f}")
    bs = brier_score(scores, test.labels)
    ece = expected_calibration_error(scores, test.labels, n_bins=10)
    report("Brier score (0 is perfect)", bs)
    report("Brier score of always predicting the base rate", base_rate * (1.0 - base_rate))
    report("expected calibration error", ece)
    passed.append(verdict("expected calibration error", ece, 0.05))
    passed.append(
        verdict("Brier score vs the base-rate predictor", bs, base_rate * (1.0 - base_rate))
    )
    print("    A calibrated probability is what makes the threshold in 5f meaningful: it is")
    print("    an operating point on a measured curve, not a knob with arbitrary units.")

    print()
    banner("VALIDATION 5e: frame-level rates, learned ranker against the Pyramid rule")
    print("    identical candidate lists; only the decision differs")
    cat6, table6 = tables[6.0]
    cases = [
        ("clean sky, sigma 5", 5.0, 0, None, 6.0),
        ("sigma 40, no false stars", 40.0, 0, None, 6.0),
        ("4 false stars", 5.0, 4, None, 6.0),
        ("8 false stars", 5.0, 8, None, 6.0),
        ("12 false stars", 5.0, 12, None, 6.0),
        ("gate 12x too wide + 4 false", 5.0, 4, 60.0, 6.0),
    ]
    summary = []
    for label, sigma, n_false, tol_sigma, mag in cases:
        cat, table = tables[mag]
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=sigma, n_false_stars=n_false)
        point = run_trials(
            cat, table, cam, cfg, BENCH_TRIALS, SEED + 20, ranker=ranker,
            thresholds=THRESHOLDS, tolerance_sigma_arcsec=tol_sigma, with_attitude=False,
        )
        summary.append((label, point))
        print(f"\n  {label}: ceiling {point.ceiling:.4f}, "
              f"{point.mean_candidates:.1f} candidates/frame, "
              f"{point.mean_seconds_per_frame * 1000:.0f} ms/frame")
        print("  " + RATE_HEADER)
        for name, result in point.methods.items():
            if name in ("triangle", "pyramid") or name in ("ranker@0.5", "ranker@0.9"):
                print("  " + rate_row(name, result.n_correct, result.n_false,
                                      result.n_none, result.n_trials))

    print()
    print("  summary at threshold 0.5")
    print(f"  {'case':<30} {'ceiling':>8} {'pyr id':>8} {'pyr fID':>8} {'ML id':>8} "
          f"{'ML fID':>8} {'id gain':>9}")
    gains = []
    for label, point in summary:
        y = point.methods["pyramid"]
        m = point.methods["ranker@0.5"]
        gain = m.identification_rate - y.identification_rate
        gains.append(gain)
        print(f"  {label:<30} {point.ceiling:8.4f} {y.identification_rate:8.4f} "
              f"{y.false_identification_rate:8.4f} {m.identification_rate:8.4f} "
              f"{m.false_identification_rate:8.4f} {gain:+9.4f}")
    print("    On a clean sky both rules are at 1.000 and there is nothing to gain. The")
    print("    ranker earns its place only where the Pyramid rule's uniqueness test throws")
    print("    away a frame it could have identified.")
    passed.append(
        verdict("identification gain on the hardest false-star case", max(gains), 0.05, mode=">=")
    )
    worst_ml_false = max(p.methods["ranker@0.5"].false_identification_rate for _, p in summary)
    worst_pyr_false = max(p.methods["pyramid"].false_identification_rate for _, p in summary)
    report("worst learned false-identification rate at threshold 0.5", worst_ml_false)
    report("worst pyramid false-identification rate over the same cases", worst_pyr_false)

    print()
    banner("VALIDATION 5f: the threshold curve")
    print("    the classical rule is one point; the ranker is a curve, and the curve is the")
    print("    result. Case: 8 false stars, sigma 5 arcsec.")
    cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=8)
    point = run_trials(
        cat6, table6, cam, cfg, 300, SEED + 21, ranker=ranker,
        thresholds=(0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99),
        with_attitude=False,
    )
    print(f"    ceiling {point.ceiling:.4f}, {point.n_trials} trials")
    print(f"  {'rule':<16} {'ident':>8} {'false ID':>9} {'none':>8} "
          f"{'false ID 95% CI':>22}")
    for name, result in point.methods.items():
        lo, hi = result.false_identification_ci
        print(f"  {name:<16} {result.identification_rate:8.4f} "
              f"{result.false_identification_rate:9.4f} {result.no_solution_rate:8.4f} "
              f"      [{lo:.4f}, {hi:.4f}]")
    pyr = point.methods["pyramid"]
    dominating = [
        n for n, r in point.methods.items()
        if n.startswith("ranker@")
        and r.identification_rate > pyr.identification_rate
        and r.false_identification_rate <= pyr.false_identification_ci[1]
    ]
    print("    thresholds beating the Pyramid rule's identification rate without exceeding")
    print(f"    the upper end of its own false-ID interval: {dominating}")
    passed.append(verdict("at least one threshold dominates the Pyramid rule",
                          float(len(dominating)), 1.0, mode=">="))

    print()
    banner("VALIDATION 5g: runtime")
    print("    the classical rule stops at the first confirmed triple; the ranker needs the")
    print("    whole candidate list, so it pays for the full scan. Measured over 120 frames.")
    tol = separation_tolerance(5.0)
    search = SearchConfig()
    print(f"  {'false stars':>12} {'early exit [ms]':>16} {'full scan [ms]':>16} "
          f"{'score [ms]':>12} {'ratio':>7}")
    for n_false in (0, 4, 8):
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=n_false)
        rng = np.random.default_rng(SEED + 22)
        scenes = [simulate_scene(cat6, cfg, rng) for _ in range(120)]
        t0 = time.perf_counter()
        for sc in scenes:
            gather_candidates(sc.vectors, sc.magnitudes, table6, tol, cam, search, True)
        t_early = (time.perf_counter() - t0) / len(scenes) * 1000.0
        t0 = time.perf_counter()
        pools = [
            gather_candidates(sc.vectors, sc.magnitudes, table6, tol, cam, search)[0]
            for sc in scenes
        ]
        t_full = (time.perf_counter() - t0) / len(scenes) * 1000.0
        t0 = time.perf_counter()
        for pool in pools:
            ranker.score_candidates(pool)
        t_score = (time.perf_counter() - t0) / len(scenes) * 1000.0
        print(f"  {n_false:12d} {t_early:16.2f} {t_full:16.2f} {t_score:12.2f} "
              f"{(t_full + t_score) / t_early:7.2f}")
    print("    The learned path is slower per frame, not faster. Its argument is the")
    print("    identification rate in 5e, not throughput.")

    print()
    banner("VALIDATION 5h: where the ranker fails")
    print("    confidence of the accepted candidate, split by whether it was correct")
    cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=10)
    point = run_trials(
        cat6, table6, cam, cfg, 300, SEED + 23, ranker=ranker,
        thresholds=(0.05,), with_attitude=False,
    )
    m = point.methods["ranker@0.05"]
    if m.confidences_correct:
        report("median confidence when correct", float(np.median(m.confidences_correct)))
    if m.confidences_false:
        report("median confidence when wrong", float(np.median(m.confidences_false)))
        report("max confidence when wrong", float(np.max(m.confidences_false)))
        print(f"    {len(m.confidences_false)} wrong acceptances out of {m.n_trials} trials at")
        print("    a threshold of 0.05; a high-confidence wrong answer is the failure that")
        print("    matters, and the maximum above is how bad it gets on these frames.")
    else:
        print("    no wrong acceptances at threshold 0.05 in this run")
    print()
    print("    Ceiling-limited frames: the ranker cannot identify what the search never")
    print(f"    proposed. Ceiling here is {point.ceiling:.4f}, learned identification rate is")
    print(f"    {m.identification_rate:.4f}, so the shortfall attributable to the decision")
    print(f"    rule is {point.ceiling - m.identification_rate:.4f} and the rest is search.")

    return finish(passed)


if __name__ == "__main__":
    raise SystemExit(main())
