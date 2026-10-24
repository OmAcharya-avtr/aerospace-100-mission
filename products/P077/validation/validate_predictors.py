"""The AI component against the closed form and against a measured-quantile lookup.

The specification for this product predicts in writing that the analytic model
wins wherever its assumptions hold and that the learned model earns its place
only in the correlated-outage regime, if it does at all. This script measures
that and reports whichever way it comes out. A baseline win is research
evidence and is not retuned away.

Four predictors, scored on held-out scenarios from a disjoint seed range:

``analytic``        the closed form, evaluated at the 0.50 quantile for its
                    point prediction and divided by the link availability to
                    extend it past its own assumptions. Sees no training data.
``analytic_mean``   the same formula with the *expectation* as its point
                    prediction instead of the median. Included because the
                    expectation is what a closed form naturally produces and
                    the median is what a point prediction should be.
``mean_lookup``     measured quantiles of the delivery time divided by the
                    clean-pass time, binned by mode, loss probability and
                    availability. What an engineer with a spreadsheet has.
``learned_quantile`` gradient-boosted quantile regression on log delivery time.

The asymmetry is stated rather than hidden: the learned model and the lookup
table both see the generator through the training set and the analytic
predictor does not. That favours the learned model, and a win for it should be
read with that in mind.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, environment_lines, relative

bootstrap()

import numpy as np  # noqa: E402

from relfiledelivery.benchmark import (  # noqa: E402
    best_on,
    calibration_table,
    coverage_standard_error,
    render_table,
    run_benchmark,
)
from relfiledelivery.dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    REGIMES,
    TARGET_COLUMN,
    fit_split,
    generate_table,
)
from relfiledelivery.predictors import (  # noqa: E402
    LearnedQuantilePredictor,
    MeanLookupPredictor,
    build_predictors,
)

out = Tee("validate_predictors_output.txt")

TRAIN_ROWS_PER_REGIME = 1500
TEST_ROWS_PER_REGIME = 600
TRAIN_SEED = 100_000
TEST_SEED = 900_000
NOMINAL = 0.90

out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("dataset")
data_timer = Timer()
with data_timer:
    raw_train = generate_table(TRAIN_ROWS_PER_REGIME, TRAIN_SEED)
with data_timer:
    raw_test = generate_table(TEST_ROWS_PER_REGIME, TEST_SEED)
train = fit_split(raw_train)
test = fit_split(raw_test)

out(f"train scenarios generated      : {len(raw_train)} (seeds from {TRAIN_SEED})")
out(f"test scenarios generated       : {len(raw_test)} (seeds from {TEST_SEED})")
out(f"train rows usable (completed)  : {len(train)}")
out(f"test rows usable (completed)   : {len(test)}")
out(f"train rows dropped             : {len(raw_train) - len(train)}")
out(f"test rows dropped              : {len(raw_test) - len(test)}")
out(f"features                       : {len(FEATURE_COLUMNS)}")
out(f"target                         : {TARGET_COLUMN}, seconds")
out("")
out("Every learned prediction here is conditional on completion: a scenario whose")
out("run did not finish inside its declared horizon has no delivery time and is")
out("dropped. The drop counts are above and the censoring is recorded in")
out("MODEL_CARD.md as a limitation, not as a footnote.")
out("")
out("completion rate by regime:")
for regime in REGIMES:
    rows = raw_test[raw_test["regime"] == regime]
    out(f"  {regime:<22} : {float(rows['completed'].mean()):.4f} of {len(rows)} scenarios")
out("")
out("delivery-time quantiles of the test target by regime, seconds:")
for regime in REGIMES:
    rows = test[test["regime"] == regime][TARGET_COLUMN]
    out(
        f"  {regime:<22} : q0.05 {float(np.quantile(rows, 0.05)):9.4f}  "
        f"q0.50 {float(np.quantile(rows, 0.50)):9.4f}  "
        f"q0.95 {float(np.quantile(rows, 0.95)):9.4f}  "
        f"max {float(np.max(rows)):9.4f}"
    )

out.rule("benchmark")
predictors = build_predictors()
fit_timer = Timer()
with fit_timer:
    rows = run_benchmark(predictors, train, test, nominal=NOMINAL)
out(f"nominal interval mass          : {NOMINAL:.2f}")
out(
    f"coverage standard error        : {coverage_standard_error(NOMINAL, len(test)):.4f} "
    f"on {len(test)} rows (all), "
    f"{coverage_standard_error(NOMINAL, len(test) // 2):.4f} per regime"
)
out("")
for line in render_table(rows):
    out(line)

out.rule("the specification's prediction, tested")
by_key = {(row.predictor, row.regime): row for row in rows}
verdicts: list[str] = []

for regime in ("no_outage", "correlated_outage"):
    analytic = by_key[("analytic", regime)]
    learned = by_key[("learned_quantile", regime)]
    lookup = by_key[("mean_lookup", regime)]
    winner = best_on(rows, regime)
    ratio = learned.median_abs_rel_error / analytic.median_abs_rel_error
    out("")
    out(f"[{regime}]")
    out(f"  analytic         median |rel err| : {analytic.median_abs_rel_error:.6f}")
    out(f"  learned_quantile median |rel err| : {learned.median_abs_rel_error:.6f}")
    out(f"  mean_lookup      median |rel err| : {lookup.median_abs_rel_error:.6f}")
    out(f"  learned / analytic                : {ratio:.4f}")
    out(f"  lowest median |rel err|           : {winner}")
    if ratio < 1.0:
        out(f"  learned beats analytic by a factor of {1.0 / ratio:.4f}")
    else:
        out(f"  analytic beats learned by a factor of {ratio:.4f}")
    verdicts.append(f"{regime}: {winner}")

out("")
out("The specification predicted: the analytic model wins where its assumptions")
out("hold, the learned model earns its place only in the correlated-outage")
out("regime, if at all. Measured above:")
for verdict in verdicts:
    out(f"  {verdict}")

out.rule("the expectation is a worse point predictor than the median")
out("Both rows below come from the same formula; they differ only in which")
out("summary of the closed-form distribution is used as the point prediction.")
out("")
header = f"{'regime':>20}{'analytic (median)':>20}{'analytic_mean':>16}{'ratio':>9}"
out(header)
out("-" * len(header))
for regime in ("all", "no_outage", "correlated_outage"):
    median_row = by_key[("analytic", regime)]
    mean_row = by_key[("analytic_mean", regime)]
    out(
        f"{regime:>20}{median_row.median_abs_rel_error:>20.6f}"
        f"{mean_row.median_abs_rel_error:>16.6f}"
        f"{mean_row.median_abs_rel_error / median_row.median_abs_rel_error:>9.4f}"
    )
out("")
out("This is the predictor-level version of the claim that a mean delivery time")
out("over an intermittent link is close to meaningless: the same model scored as")
out("a mean is worse than the same model scored as a median, on the same rows.")

out.rule("is the uncertainty output calibrated")
out("Coverage is the fraction of held-out rows whose observed delivery time fell")
out("inside the predicted interval. An interval that is not calibrated is")
out("reported as not calibrated. Width is the median of (upper - lower) divided")
out("by the observed time: coverage without a width is not a result, because an")
out("interval from zero to infinity covers everything.")
out("")
header = (
    f"{'predictor':>18}{'regime':>20}{'nominal':>9}{'coverage':>10}"
    f"{'error':>9}{'std err':>9}{'width/obs':>11}{'calibrated':>12}"
)
out(header)
out("-" * len(header))
calibration_failures: list[str] = []
for row in rows:
    error = coverage_standard_error(row.nominal_coverage, row.rows)
    if not row.is_calibrated:
        calibration_failures.append(
            f"{row.predictor}/{row.regime}: coverage {row.coverage:.4f} against nominal "
            f"{row.nominal_coverage:.2f}"
        )
    out(
        f"{row.predictor:>18}{row.regime:>20}{row.nominal_coverage:>9.2f}"
        f"{row.coverage:>10.4f}{row.calibration_error:>+9.4f}{error:>9.4f}"
        f"{row.median_interval_width_ratio:>11.4f}{str(row.is_calibrated):>12}"
    )
out("")
out(f"rows outside the declared 0.03 calibration band : {len(calibration_failures)} of {len(rows)}")
for failure in calibration_failures:
    out(f"  NOT CALIBRATED {failure}")
out("")
out("The calibration band is 0.03, declared in benchmark.py before the")
out("measurement: roughly three standard errors of a coverage estimate at a")
out("nominal 0.9 on a thousand rows.")

out.rule("calibration curve: measured coverage at three nominal levels")
out("")
header = f"{'predictor':>18}{'nominal':>9}{'coverage':>10}{'width/obs':>11}"
out(header)
out("-" * len(header))
for predictor in predictors:
    for nominal, coverage, width in calibration_table(predictor, test, (0.50, 0.80, 0.90)):
        out(f"{predictor.name:>18}{nominal:>9.2f}{coverage:>10.4f}{width:>11.4f}")

out.rule("the learned model, described")
learned = next(p for p in predictors if isinstance(p, LearnedQuantilePredictor))
lookup = next(p for p in predictors if isinstance(p, MeanLookupPredictor))
out(f"training rows                  : {learned.n_train}")
out(f"quantile levels fitted         : {list(learned.levels)}")
out(f"trees per level                : {learned.n_estimators}")
out(f"max depth                      : {learned.max_depth}")
out(f"learning rate                  : {learned.learning_rate}")
out(f"subsample                      : {learned.subsample}")
out(f"random state                   : {learned.random_state}")
learned.predict_quantiles(test, learned.levels)
pairs = len(test) * (len(learned.levels) - 1)
out(
    f"quantile crossings sorted away : {learned.crossing_count} of {pairs} adjacent "
    f"level pairs ({learned.crossing_count / pairs:.4f})"
)
out("")
out("A crossing is an adjacent pair of levels on one row where the higher-quantile")
out("model predicted a lower value than the lower-quantile model. Seven levels")
out("give six adjacent pairs per row. Independent per-level fits do not guarantee")
out("monotonicity, so the predictions are sorted across levels and the number")
out("sorted is counted rather than hidden. A crossing rate of this size says the")
out("per-level fits disagree about the shape of the conditional distribution in")
out("part of the feature space, which is consistent with the interval")
out("under-covering: it is not a cosmetic detail.")
out("")
lookup.predict(test)
out(f"lookup bins used               : {lookup.lookup_count} of {len(test)} rows")
out(f"lookup fell back to the global table on : {lookup.fallback_count} rows")
out("")
out("impurity-based feature importances of the median model, descending:")
importances = learned.feature_importances(0.50)
for name, value in sorted(importances.items(), key=lambda item: -item[1]):
    out(f"  {name:<30} {value:.6f}")
out("")
out("Impurity importances are biased towards high-cardinality features (Strobl")
out("et al. 2007) and describe the fitted model, not the link.")

out.rule("compute budget (UNSTABLE: wall clock on a shared two-core container)")
out("These numbers move between runs and are not used in any conclusion.")
out("")
out(f"{'stage':>28}{'seconds':>10}")
out("-" * 38)
out(f"{'scenario generation':>28}{data_timer.elapsed_s:>10.2f}")
out(f"{'fit and score 4 predictors':>28}{fit_timer.elapsed_s:>10.2f}")
out(f"{'total':>28}{data_timer.elapsed_s + fit_timer.elapsed_s:>10.2f}")

out.rule("sanity checks on every prediction")
out("Measured, not assumed. A predictor that emitted a negative or non-finite")
out("delivery time, or an interval that did not contain its own point")
out("prediction, would score perfectly well on a median relative error and be")
out("nonsense.")
out("")
out("The assertion covers the three predictors whose point prediction is the")
out("0.50 quantile of the same expression that produces their interval, so")
out("containment is a property of the construction. analytic_mean is excluded")
out("and reported separately: its point prediction is the EXPECTATION and its")
out("interval ends are quantiles, and those are two different summaries of the")
out("same distribution with no containment relation between them. The first run")
out("of this check asserted containment for all four and found analytic_mean")
out("violating it. That is a finding, not a defect to clamp away, and clamping")
out("the expectation into its own interval would have hidden it.")
out("")
header = (
    f"{'predictor':>18}{'rows':>7}{'finite':>8}{'positive':>10}"
    f"{'lo<=point':>11}{'point<=hi':>11}{'asserted':>10}{'ok':>5}"
)
out(header)
out("-" * len(header))
sane = True
mean_outside = 0
mean_worst_excess = 0.0
for predictor in predictors:
    lower, median, upper = predictor.predict_interval(test, nominal=NOMINAL)
    finite = int(np.sum(np.isfinite(median) & np.isfinite(lower) & np.isfinite(upper)))
    positive = int(np.sum(median > 0.0))
    lo_ok = int(np.sum(lower <= median + 1e-9))
    hi_ok = int(np.sum(median <= upper + 1e-9))
    ok = finite == positive == lo_ok == hi_ok == len(test)
    asserted = predictor.name != "analytic_mean"
    if asserted:
        sane = sane and ok
    else:
        outside = median > upper + 1e-9
        mean_outside = int(np.sum(outside))
        if mean_outside:
            mean_worst_excess = float(np.max((median[outside] - upper[outside]) / upper[outside]))
    out(
        f"{predictor.name:>18}{len(test):>7}{finite:>8}{positive:>10}"
        f"{lo_ok:>11}{hi_ok:>11}{str(asserted):>10}{str(ok):>5}"
    )
out("")
out(
    f"analytic_mean point prediction above its own 0.95 end on : {mean_outside} of "
    f"{len(test)} rows"
)
out(f"worst excess over the upper end                          : {mean_worst_excess:.6f} relative")
out("")
out("The closed-form expectation can exceed the closed-form 0.95 quantile")
out("because the delivery-time distribution over an intermittent link has a tail")
out("heavy enough to pull the mean past the ninety-fifth percentile of the")
out("semi-analytic approximation. It is the same phenomenon as the mean/median")
out("ratio reported in validate_distributions.py, seen from the predictor side.")

out.rule("summary")
checks = {
    "the analytic predictor beats the lookup table where it is exact": (
        by_key[("analytic", "no_outage")].median_abs_rel_error
        < by_key[("mean_lookup", "no_outage")].median_abs_rel_error
    ),
    "every prediction is finite, positive and bracketed by its interval": sane,
}
reported = {
    "learned_quantile beats analytic in the correlated-outage regime": (
        by_key[("learned_quantile", "correlated_outage")].median_abs_rel_error
        < by_key[("analytic", "correlated_outage")].median_abs_rel_error
    ),
    "learned_quantile beats analytic in the no-outage regime": (
        by_key[("learned_quantile", "no_outage")].median_abs_rel_error
        < by_key[("analytic", "no_outage")].median_abs_rel_error
    ),
    "the learned interval is calibrated at nominal 0.90 (all rows)": by_key[
        ("learned_quantile", "all")
    ].is_calibrated,
    "the analytic interval is calibrated at nominal 0.90 (all rows)": by_key[
        ("analytic", "all")
    ].is_calibrated,
}
out("asserted checks:")
for name, passed in checks.items():
    out(f"  {'PASS' if passed else 'FAIL'}  {name}")
out("")
out("reported outcomes -- these are measurements, not pass/fail criteria, and")
out("are recorded in whichever direction they came out:")
for name, value in reported.items():
    out(f"  {'TRUE ' if value else 'FALSE'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_predictors.py: at least one asserted check FAILED")
