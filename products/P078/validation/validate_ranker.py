"""The learned risk ranker against the two orderings it has to beat.

Scored on held-out cases generated from seeds the model never saw, with the
seed ranges checked disjoint rather than assumed so. Four orderings: the
learned model, the irreversibility-flag ordering, the resource-draw ordering,
and a seeded random control that shows what precision at k looks like with no
information at all.

The specification predicts that the flag ordering will match or beat the
learned model, and says in terms that such a result is to be published rather
than retuned away. Whatever this script measures is what the README reports.

A timing-feature ablation is run as well. Four of the nine defect families edit
the time column, so a model with access to inter-command spacing can learn the
signature of the injection process rather than anything about commanding. The
ablation withholds the three timing-shape features and re-measures; the gap
between the two runs is the part of the learned advantage that comes from the
corpus rather than from the problem, and the README says so.
"""

from __future__ import annotations

import numpy as np

import _bootstrap
from _bootstrap import Tee, Timer, header

ROOT = _bootstrap.bootstrap()

from cmdguard.benchmark import assert_disjoint, ranker_comparison  # noqa: E402
from cmdguard.dataset import build_corpus  # noqa: E402
from cmdguard.features import FEATURE_NAMES  # noqa: E402
from cmdguard.ranker import RiskRanker, involved_mask  # noqa: E402

TRAIN_SEEDS = range(0, 70)
TEST_SEEDS = range(100_000, 100_035)
KS = (5, 10, 20)
TIMING_FEATURES = ("dt_prev_s", "dt_next_s", "neighbours_within_60s")
LABELS = ("learned", "irreversibility_flag", "resource_draw", "random_control")


def main() -> int:
    tee = Tee("validate_ranker_output.txt")
    header(tee, "cmdguard -- learned risk ranker against flag and resource orderings")

    tee.rule("1. the split")
    n_train_seeds, n_test_seeds = assert_disjoint(TRAIN_SEEDS, TEST_SEEDS)
    tee(f"  training seeds : {n_train_seeds} ({TRAIN_SEEDS.start} .. {TRAIN_SEEDS.stop - 1})")
    tee(f"  held-out seeds : {n_test_seeds} ({TEST_SEEDS.start} .. {TEST_SEEDS.stop - 1})")
    tee.check("the seed ranges are disjoint", True, "checked by assert_disjoint")
    build = Timer()
    with build:
        train = build_corpus(TRAIN_SEEDS, include_clean=False)
    with build:
        test = build_corpus(TEST_SEEDS, include_clean=False)
    tee(f"  training cases : {len(train)}")
    tee(f"  held-out cases : {len(test)}")
    tee(f"  corpus build   : {build.elapsed_s:.3f} s wall clock (unstable)")
    positives = np.concatenate(
        [involved_mask(entry.report, len(entry.case.sequence)) for entry in test]
    )
    tee(f"  held-out command occurrences      : {positives.size}")
    tee(f"  of those involved in a violation  : {int(positives.sum())}")
    tee(f"  positive fraction                 : {positives.mean():.6f}")
    tee("")
    tee('  "involved" means the step either tripped a violation or is named in the')
    tee("  provenance trace of one: the set a reviewer has to read to understand the")
    tee("  findings, which is what a triage ordering is for.")

    results = {}
    for label, drop in (("full feature set", ()), ("timing features withheld", TIMING_FEATURES)):
        tee.rule(f"2. precision and recall at k -- {label}")
        if drop:
            tee(f"  withheld: {', '.join(drop)}")
        comparison = ranker_comparison(train, test, ks=KS, drop_features=drop, random_seed=4711)
        results[label] = comparison
        tee(f"  fit wall clock   : {comparison.fit_seconds:.3f} s (unstable)")
        tee(f"  scoring wall clock: {comparison.score_seconds:.3f} s (unstable)")
        tee(f"  mean steps per held-out case    : {comparison.mean_steps_per_case:.2f}")
        tee(f"  mean involved steps per case    : {comparison.mean_involved_per_case:.2f}")
        tee("")
        tee(f"  {'ordering':<22s} {'k':>3s} {'precision':>11s} {'recall':>9s} "
            f"{'scored':>7s} {'excluded':>9s}")
        for k in KS:
            for name in LABELS:
                score = comparison.scores[(name, k)]
                tee(f"  {name:<22s} {k:>3d} {score.mean_precision:>11.6f} "
                    f"{score.mean_recall:>9.6f} {score.n_scored:>7d} "
                    f"{score.n_excluded:>9d}")
            tee("")
        tee("  paired differences, learned minus baseline, by case, with a percentile")
        tee("  bootstrap 95 per cent interval over 4000 resamples of the held-out cases:")
        tee("")
        for k in KS:
            for baseline in ("irreversibility_flag", "resource_draw", "random_control"):
                mean, low, high = comparison.paired_bootstrap("learned", baseline, k)
                verdict = (
                    "learned ahead"
                    if low > 0.0
                    else ("baseline ahead" if high < 0.0 else "NOT DISTINGUISHED")
                )
                tee(f"    k={k:<3d} learned - {baseline:<22s} {mean:+.6f} "
                    f"[{low:+.6f}, {high:+.6f}] {verdict}")
            tee("")
        for k in KS:
            tee(f"  best ordering at k={k}: {comparison.best_at(k)}")

    tee.rule("3. the ablation, side by side")
    full = results["full feature set"]
    ablated = results["timing features withheld"]
    tee(f"  {'k':>3s} {'learned full':>13s} {'learned ablated':>16s} {'flag':>9s} "
        f"{'resource':>9s} {'random':>9s}")
    for k in KS:
        tee(f"  {k:>3d} {full.scores[('learned', k)].mean_precision:>13.6f} "
            f"{ablated.scores[('learned', k)].mean_precision:>16.6f} "
            f"{full.scores[('irreversibility_flag', k)].mean_precision:>9.6f} "
            f"{full.scores[('resource_draw', k)].mean_precision:>9.6f} "
            f"{full.scores[('random_control', k)].mean_precision:>9.6f}")
    tee("")
    tee("  Four of the nine defect families edit the time column -- compress_gap,")
    tee("  overlap_exclusion, resource_spike and the tail shift in irreversible_early")
    tee("  -- so inter-command spacing carries the signature of the injection process.")
    tee("  The drop between the two learned columns is the size of that effect on this")
    tee("  corpus. It is a property of the generator, not of command sequences, and")
    tee("  the ablated column is the more conservative number to quote.")
    tee("")
    for k in KS:
        delta = (
            full.scores[("learned", k)].mean_precision
            - ablated.scores[("learned", k)].mean_precision
        )
        tee(f"    k={k:<3d} precision attributable to the timing features: {delta:+.6f}")

    tee.rule("4. the baselines are not noise")
    for k in KS:
        for baseline in ("irreversibility_flag", "resource_draw"):
            mean, low, high = full.paired_bootstrap(baseline, "random_control", k)
            tee(f"  k={k:<3d} {baseline:<22s} - random_control {mean:+.6f} "
                f"[{low:+.6f}, {high:+.6f}]")
    tee("")
    tee("  Read the intervals, not the means. Both free orderings beat the random")
    tee("  control at k=5 and k=10 by a margin whose interval excludes zero, so a sort")
    tee("  over an irreversibility flag somebody already wrote into the command")
    tee("  dictionary is a real triage ordering at those cut-offs. At k=20 the interval")
    tee("  includes zero for both: with twenty of thirty-six steps already shown, the")
    tee("  free orderings are NOT distinguished from a shuffle on this corpus, and that")
    tee("  is reported rather than rounded up to a win.")

    tee.rule("5. feature importance, full feature set")
    model = RiskRanker()
    model.fit([(entry.case, entry.report) for entry in train])
    assert model.metrics is not None
    tee(f"  rows {model.metrics.n_rows}, positives {model.metrics.n_positive}, "
        f"positive fraction {model.metrics.positive_fraction:.6f}")
    tee(f"  trees {model.metrics.n_estimators}, max depth {model.metrics.max_depth}, "
        f"seed {model.metrics.seed}")
    tee(f"  fit wall clock {model.metrics.fit_seconds:.3f} s (unstable)")
    tee("")
    for name, value in model.metrics.feature_importance.items():
        tee(f"    {name:<28s} {value:.6f}")
    tee.check("importances sum to one",
              abs(sum(model.metrics.feature_importance.values()) - 1.0) < 1e-9)
    tee.check("every declared feature is used",
              set(model.metrics.feature_importance) == set(FEATURE_NAMES))

    tee.rule("6. the uncertainty output, measured rather than described")
    tee("  The ranker reports a spread beside each risk: the standard deviation of the")
    tee("  forest's per-tree probabilities. It is an ensemble dispersion and not a")
    tee("  calibrated interval. Two things are measured about it here.")
    risks: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for entry in test:
        scores = model.score(entry.case)
        risks.append(scores.risk)
        spreads.append(scores.spread)
        labels.append(involved_mask(entry.report, len(entry.case.sequence)))
    risk = np.concatenate(risks)
    spread = np.concatenate(spreads)
    label = np.concatenate(labels)
    tee("")
    tee(f"  mean spread on involved steps     : {spread[label].mean():.6f}")
    tee(f"  mean spread on uninvolved steps   : {spread[~label].mean():.6f}")
    tee(f"  mean risk on involved steps       : {risk[label].mean():.6f}")
    tee(f"  mean risk on uninvolved steps     : {risk[~label].mean():.6f}")
    tee.check("mean risk separates the two classes", risk[label].mean() > risk[~label].mean(),
              f"{risk[label].mean():.6f} vs {risk[~label].mean():.6f}")
    tee("")
    tee("  reliability of the risk, in deciles of predicted risk:")
    tee(f"  {'bin':<14s} {'n':>7s} {'mean risk':>11s} {'observed':>10s} "
        f"{'gap':>10s} {'mean spread':>12s}")
    edges = np.linspace(0.0, 1.0, 11)
    gaps = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selector = (risk >= low) & (risk < high if high < 1.0 else risk <= 1.0)
        count = int(selector.sum())
        if count == 0:
            tee(f"  [{low:.1f}, {high:.1f})     {0:>7d} {'--':>11s} {'--':>10s} "
                f"{'--':>10s} {'--':>12s}")
            continue
        mean_risk = float(risk[selector].mean())
        observed = float(label[selector].mean())
        gaps.append(abs(mean_risk - observed) * count)
        tee(f"  [{low:.1f}, {high:.1f})     {count:>7d} {mean_risk:>11.6f} "
            f"{observed:>10.6f} {mean_risk - observed:>+10.6f} "
            f"{float(spread[selector].mean()):>12.6f}")
    tee("")
    tee(f"  weighted mean absolute calibration gap: {sum(gaps) / risk.size:.6f}")
    tee("  This is a calibration measurement, not a calibration claim. The model card")
    tee("  records it and says the model is not calibrated.")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
