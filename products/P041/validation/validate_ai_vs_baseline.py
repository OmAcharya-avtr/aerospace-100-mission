"""AI against the analytic baseline, on held-out seeds. Honest either way.

Phase order, which is not negotiable: the analytic level-crossing-rate /
mean-fade-duration result is implemented and validated first
(``validate_crossing_convergence.py``), then closed with the memoryless
assumption of equation (16) to give an exceedance probability, then benchmarked
against the learned model on the same held-out data.

Three predictors are compared on identical held-out events:

``analytic_configured``
    ``exp(-t/MFD)`` with the **exact** sampled-Gauss-Markov MFD computed from the
    true scintillation index and correlation time. State-blind. Given the channel
    configuration that the learned model is not given.
``empirical_constant``
    The measured exceedance frequency on the training paths. Also state-blind, and
    the same analytic structure with the exponential assumption replaced by a
    measurement. **This is the baseline that matters**: beating a miscalibrated
    constant proves nothing about state.
``learned``
    The bagged-tree model, with an uncertainty output.

Metrics: Brier score, log loss, ROC AUC, expected calibration error, plus a
reliability table for the learned model and a report of whether its uncertainty
output separates the cases it gets wrong.

Run: ``PYTHONPATH=../src python3 validate_ai_vs_baseline.py``
Runtime: about 100 s on one core.
"""

from __future__ import annotations

import time

import numpy as np

from codedfade.fade import required_interleaver_depth
from codedfade.predictor import (
    FEATURE_NAMES,
    FadeExceedancePredictor,
    brier_score,
    build_dataset,
    empirical_exceedance_baseline,
    expected_calibration_error,
    log_loss_score,
    reliability_table,
    roc_auc,
)

SI, TAU, FS = 0.6, 2.0e-4, 1.0e6
THRESHOLD = 0.6
TARGET = 14
SAMPLES = 200_000
TRAIN_SEEDS = list(range(24))
TEST_SEEDS = list(range(100, 112))


def main() -> None:
    print("=" * 88)
    print("AI VERSUS ANALYTIC BASELINE: FADE-DURATION EXCEEDANCE")
    print("=" * 88)
    print(f"channel                  lognormal, SI={SI}, tau={TAU:.1e} s, "
          f"fs={FS:.1e} Hz, kernel=exp")
    print(f"threshold amplitude      {THRESHOLD}")
    print(f"target duration          {TARGET} samples = {TARGET / FS:.3e} s")
    print(f"samples per path         {SAMPLES}")
    print(f"train seeds              {TRAIN_SEEDS[0]}..{TRAIN_SEEDS[-1]} "
          f"({len(TRAIN_SEEDS)} paths)")
    print(f"test seeds               {TEST_SEEDS[0]}..{TEST_SEEDS[-1]} "
          f"({len(TEST_SEEDS)} paths, disjoint from train)")
    print("split strategy           by path seed, so no event from a training")
    print("                         realisation appears in the test set")
    print()

    t0 = time.perf_counter()
    train = build_dataset(TRAIN_SEEDS, SAMPLES, THRESHOLD, TARGET, SI, TAU, FS)
    test = build_dataset(TEST_SEEDS, SAMPLES, THRESHOLD, TARGET, SI, TAU, FS)
    print(f"train events             {train.features.shape[0]}  "
          f"exceedance frequency {float(train.labels.mean()):.6f}")
    print(f"test events              {test.features.shape[0]}  "
          f"exceedance frequency {float(test.labels.mean()):.6f}")
    print(f"dataset build            {time.perf_counter() - t0:.1f} s")
    print()

    t0 = time.perf_counter()
    model = FadeExceedancePredictor(160, 8, 0).fit(train.features, train.labels)
    print(f"forest fit               {time.perf_counter() - t0:.1f} s "
          f"(160 trees, max_depth 8, min_samples_leaf 20)")
    out = model.predict(test.features)
    print()

    analytic = test.analytic_configured
    observed_feature = test.features[
        :, FEATURE_NAMES.index("analytic_observed_exceedance")
    ]
    empirical = np.full(test.labels.size, empirical_exceedance_baseline(train.labels))
    observed = float(test.labels.mean())

    print("-- the analytic closure, measured ------------------------------------")
    print(f"eq(16) prediction        P(T > MFD) = {float(analytic[0]):.6f}")
    print(f"observed frequency       {observed:.6f}")
    print(f"ratio predicted/observed {float(analytic[0]) / observed:.4f}")
    print()
    print("The memoryless closure of the level-crossing result over-predicts the")
    print("exceedance at t = MFD by the ratio above. Fade durations on this channel")
    print("are not exponentially distributed: short fades are far more common than an")
    print("exponential with the same mean. This is published as measured; equation")
    print("(16) is not retuned and not removed.")
    print()

    print("-- benchmark on identical held-out events ----------------------------")
    header = f"{'predictor':<24} {'Brier':>10} {'log loss':>10} {'ROC AUC':>9} {'ECE':>9}"
    print(header)
    print("-" * len(header))
    rows = [
        ("analytic_configured", analytic),
        ("analytic_observed (feat)", observed_feature),
        ("empirical_constant", empirical),
        ("learned", out.probability),
    ]
    scores = {}
    for name, p in rows:
        b = brier_score(p, test.labels)
        scores[name] = b
        print(
            f"{name:<24} {b:10.6f} {log_loss_score(p, test.labels):10.6f} "
            f"{roc_auc(p, test.labels):9.4f} "
            f"{expected_calibration_error(p, test.labels):9.6f}"
        )
    print()
    best_baseline = min(
        scores["analytic_configured"],
        scores["empirical_constant"],
        scores["analytic_observed (feat)"],
    )
    gain = (best_baseline - scores["learned"]) / best_baseline
    print(f"best baseline Brier      {best_baseline:.6f}")
    print(f"learned Brier            {scores['learned']:.6f}")
    print(f"relative improvement     {gain:+.6f}  "
          f"({'learned wins' if gain > 0 else 'BASELINE WINS'})")
    print()
    print("VERDICT")
    if gain > 0:
        print("  The learned model improves on the best state-blind baseline, and the")
        print("  ROC AUC above 0.5 is the direct evidence that the state at the")
        print("  crossing carries information about the fade length. The size of the")
        print("  improvement over the empirical constant is the number to quote; the")
        print("  improvement over the exponential closure is mostly a calibration")
        print("  correction and overstates the case.")
    else:
        print("  The analytic baseline wins. That is the published result.")
    print()

    print("-- reliability table, learned model ----------------------------------")
    mean_p, obs, count = reliability_table(out.probability, test.labels, bins=10)
    print(f"{'bin':>5} {'mean pred':>11} {'observed':>10} {'count':>8} {'diff':>10}")
    for b in range(10):
        if count[b] == 0:
            print(f"{b:5d} {'-':>11} {'-':>10} {0:8d} {'-':>10}")
            continue
        print(
            f"{b:5d} {mean_p[b]:11.6f} {obs[b]:10.6f} {count[b]:8d} "
            f"{mean_p[b] - obs[b]:+10.6f}"
        )
    print()

    print("-- uncertainty output ------------------------------------------------")
    print(f"mean per-tree sigma      {float(out.uncertainty.mean()):.6f}")
    print(f"median                   {float(np.median(out.uncertainty)):.6f}")
    print(f"max                      {float(out.uncertainty.max()):.6f}")
    hard = np.abs(out.probability - test.labels) > 0.5
    print(f"events with |p-y| > 0.5  {int(hard.sum())} of {test.labels.size}")
    print(f"mean sigma on those      {float(out.uncertainty[hard].mean()):.6f}")
    print(f"mean sigma on the rest   {float(out.uncertainty[~hard].mean()):.6f}")
    ratio = float(out.uncertainty[hard].mean() / out.uncertainty[~hard].mean())
    print(f"ratio                    {ratio:.4f}")
    print("A ratio above 1 means the ensemble is more uncertain exactly where it is")
    print("wrong, which is what makes the uncertainty usable for gating a depth")
    print("decision. A ratio near 1 means it is not, and that is reported as such.")
    print()

    print("-- what the model is actually for: sizing a depth --------------------")
    print("Depth required to cover a fade of a given length at Rs = 1 Mbaud, with the")
    print("latency that depth costs, so the probability output has a decision attached.")
    print(f"{'quantile':>10} {'duration sym':>13} {'depth':>8} {'latency ms (span 31)':>22}")
    for q in (0.5, 0.9, 0.99, 1.0):
        d = float(np.quantile(test.durations_samples, q))
        depth = required_interleaver_depth(d / FS, FS)
        print(f"{q:10.2f} {d:13.1f} {depth:8d} {2 * depth * 31 / FS * 1000:22.3f}")
    print()
    print("-- feature importances -----------------------------------------------")
    for name, imp in sorted(
        zip(FEATURE_NAMES, model.feature_importances, strict=True),
        key=lambda kv: -kv[1],
    ):
        print(f"  {name:34s} {imp:.6f}")
    print()
    print("=" * 88)


if __name__ == "__main__":
    main()
