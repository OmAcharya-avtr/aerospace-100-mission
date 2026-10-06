"""Validation: short-horizon outage forecasting. Baselines first, then the forest.

The problem
-----------
At decision instant ``t`` the receiver has the last 400 samples of amplitude
(0.4 ms, two correlation lengths) and must say whether a fade of amplitude 0.35
begins in the next 200 samples (0.2 ms, one correlation length). Decisions are
taken every 50 samples. The base rate is about 2.6 %, which is the whole
difficulty: accuracy is near-useless at that rate, and a forecaster that merely
emits the base rate is already perfectly calibrated.

Protocol, fixed before any model was fitted
-------------------------------------------
* One 4 000 000-sample record (4 s at 1 MHz), seed 4901.
* Temporal split 60 / 15 / 25 with a 12-row gap, so the window and horizon
  either side of a split boundary are disjoint. No shuffling: a random split of
  a correlated series leaks near-duplicate rows into training.
* Three baselines implemented and scored first: the constant base rate, the
  analytic conditional level-crossing-rate predictor (which uses no labels at
  all, only a two-parameter channel fit to the training amplitude), and
  logistic regression.
* Forest hyperparameters chosen on the **calibration** split by Brier score,
  never on the test split. The selected configuration is then scored once.
* Platt recalibration of the analytic predictor and of the forest, fitted on
  the calibration split, reported as separate rows.
* Every model scored on the same test rows, with the Brier skill score measured
  against a constant forecast at the **training** base rate.

Runtime: about 100 s on two cores.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import spearmanr

from linkoutage.calibration import (
    CalibrationReport,
    brier_score,
    evaluate_forecast,
)
from linkoutage.channel import lognormal_amplitude_series
from linkoutage.features import FEATURE_NAMES, build_outage_dataset
from linkoutage.predictors import (
    AnalyticLcrPredictor,
    ConstantRatePredictor,
    LogisticBaseline,
    PlattCalibrated,
    RandomForestOutageClassifier,
)

N_SAMPLES = 4_000_000
FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
SEED = 4901
THRESHOLD = 0.35
WINDOW = 400
HORIZON = 200
STRIDE = 50
TREES = 200
LEAF_GRID = (5, 20, 60)

print("=" * 78)
print("validate_outage_classifier.py")
print("=" * 78)

series = lognormal_amplitude_series(N_SAMPLES, fs_hz=FS, tau_s=TAU, si=SI, seed=SEED)
data = build_outage_dataset(
    series.amplitude,
    threshold=THRESHOLD,
    window_samples=WINDOW,
    horizon_samples=HORIZON,
    stride_samples=STRIDE,
    gaussian=series.gaussian,
)
train, cal, test = data.split.train, data.split.calibration, data.split.test

print()
print(series.describe())
print()
print(data.report())
print()
print(f"  prediction threshold T       : {THRESHOLD!r} (amplitude)")
print(f"  window  {WINDOW} samples        : {WINDOW / FS!r} s = {WINDOW / (TAU * FS)!r} tau")
print(f"  horizon {HORIZON} samples        : {HORIZON / FS!r} s = {HORIZON / (TAU * FS)!r} tau")
print(f"  stride  {STRIDE} samples         : {STRIDE / FS!r} s")
print(f"  features                     : {len(FEATURE_NAMES)}")
print("  effective sample size is set by the number of independent fade onsets,")
print("  not by the row count: adjacent rows overlap by 350 of 400 window")
print(f"  samples. Test onsets: {int(data.y[test].sum())}.")

print()
print("-" * 78)
print("PHASE 1 -- BASELINES, implemented and scored before any learned model")
print("-" * 78)

constant = ConstantRatePredictor().fit(data.y[train])
print()
print("[B1] constant base rate")
print(f"  training base rate       : {constant.rate!r}")
print(f"  binomial standard error  : {constant.rate_standard_error()!r}")
print(f"  training rows            : {constant.n_train}, positives {constant.n_positive_train}")
print(f"  test base rate           : {float(data.y[test].mean())!r}")
print("  This forecaster is perfectly calibrated by construction and has zero")
print("  resolution. Any model whose reliability term is worse than its is")
print("  miscalibrated, whatever its AUC says.")

last_train_sample = int(data.index[train[-1]])
analytic = AnalyticLcrPredictor(threshold=THRESHOLD, horizon_samples=HORIZON).fit(
    series.amplitude[: last_train_sample + 1]
)
print()
print("[B2] analytic conditional level-crossing-rate predictor")
print(f"  fitted on amplitude samples 0 .. {last_train_sample} (training segment only)")
print(f"  {analytic.estimate.describe()}")
print(f"  true SI {SI!r}, true rho {series.rho!r}")
print(f"  Gaussian level u         : {analytic.level!r}")
print("  uses NO labels: the channel parameters come from the amplitude record")
print("  and the forecast from the exact conditional expected crossing count,")
print("  with a Poisson-clumping step from count to probability")

logistic = LogisticBaseline().fit(data.x[train], data.y[train])
print()
print("[B3] logistic regression on the 14 standardised features")
coefs = logistic.coefficients()
order = np.argsort(-np.abs(coefs))
print(f"  {'feature':<32s} {'coefficient (standardised)':>28s}")
for i in order:
    print(f"  {FEATURE_NAMES[i]:<32s} {coefs[i]:>28.6f}")

print()
print("-" * 78)
print("PHASE 2 -- LEARNED MODEL, hyperparameters selected on the CALIBRATION split")
print("-" * 78)
header = (
    f"  {'min_samples_leaf':>17s} {'trees':>6s} {'calibration Brier':>18s} "
    f"{'selected':>9s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
sweep = []
for leaf in LEAF_GRID:
    model = RandomForestOutageClassifier(
        n_estimators=TREES, min_samples_leaf=leaf, seed=SEED
    ).fit(data.x[train], data.y[train])
    bs_cal = brier_score(data.y[cal], model.predict_proba_onset(data.x[cal]))
    sweep.append((leaf, bs_cal, model))
best_leaf, best_cal, forest = min(sweep, key=lambda r: r[1])
for leaf, bs_cal, _ in sweep:
    print(
        f"  {leaf:>17d} {TREES:>6d} {bs_cal:>18.8f} "
        f"{'<-- yes' if leaf == best_leaf else '':>9s}"
    )
print(f"  selected min_samples_leaf = {best_leaf}, calibration Brier {best_cal!r}")
print("  The test split was not consulted in this choice.")

print()
print("-" * 78)
print("PHASE 3 -- HELD-OUT TEST SPLIT, every forecaster on the same rows")
print("-" * 78)
rows = []
for model in (constant, analytic, logistic, forest):
    rows.append(
        evaluate_forecast(
            data.y[test],
            model.predict_proba_onset(data.x[test]),
            name=model.name,
            reference_rate=constant.rate,
        )
    )
platt_rows = []
for base in (analytic, forest):
    platt = PlattCalibrated(base).fit(data.x[cal], data.y[cal])
    platt_rows.append(
        evaluate_forecast(
            data.y[test],
            platt.predict_proba_onset(data.x[test]),
            name=platt.name,
            reference_rate=constant.rate,
        )
    )
rows += platt_rows

print()
print(CalibrationReport.header())
print("-" * len(CalibrationReport.header()))
for row in rows:
    print(row.line())
print()
print("  Brier: lower is better. BSS: Brier skill score against a constant")
print("  forecast at the TRAINING base rate; positive means better than that")
print("  baseline, negative means worse. REL: reliability (calibration error,")
print("  lower better). RES: resolution (skill, higher better). ECE/MCE: mean")
print("  and maximum calibration gap over ten equal-count bins. AP: average")
print("  precision. mean p: mean forecast, compare against the test base rate")
print(f"  of {float(data.y[test].mean())!r}.")

best = min(rows, key=lambda r: r.brier)
print()
print(f"  LOWEST BRIER ON THE HELD-OUT SPLIT: {best.name}  ({best.brier!r})")
learned = [r for r in rows if "forest" in r.name]
baselines = [r for r in rows if "forest" not in r.name]
best_learned = min(learned, key=lambda r: r.brier)
best_baseline = min(baselines, key=lambda r: r.brier)
print(f"  best baseline  : {best_baseline.name}  Brier {best_baseline.brier!r}")
print(f"  best learned   : {best_learned.name}  Brier {best_learned.brier!r}")
if best_baseline.brier <= best_learned.brier:
    print("  A BASELINE WINS. Per ADR-011 that is the published result; the")
    print("  forest is reported in the README with these numbers and is not")
    print("  retuned to change the outcome.")
else:
    print("  The learned model wins on this split, by the margin above.")

print()
print("[C1] Why accuracy is not the headline")
print(f"  all-negative accuracy on the test split : {rows[0].accuracy_all_negative!r}")
for row in rows:
    print(f"  {row.name:<30s} accuracy at p >= 0.5 : {row.accuracy_at_half!r}")
print("  Every forecaster scores within a percent of the do-nothing accuracy.")
print("  The metric cannot distinguish them and is reported only to show that.")

print()
print("[C2] Brier decomposition, BS = REL - RES + UNC + binning residual")
header = (
    f"  {'forecaster':<30s} {'BS':>11s} {'REL':>11s} {'RES':>11s} {'UNC':>11s} "
    f"{'residual':>11s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for row in rows:
    print(
        f"  {row.name:<30s} {row.brier:>11.3e} {row.reliability:>11.3e} "
        f"{row.resolution:>11.3e} {row.uncertainty:>11.3e} "
        f"{row.decomposition_residual:>11.3e}"
    )
print("  The constant-rate row has RES = 0 exactly: that is what no skill looks")
print("  like. The residual is the binning error of the decomposition; it is")
print("  small relative to BS for every row, so the partition is usable.")

print()
print("[C3] Reliability curves, ten equal-count bins")
for row in rows:
    print()
    print(f"  {row.name}")
    print("  " + row.curve.table().replace("\n", "\n  "))

print()
print("[C4] Uncertainty output of the forest")
mean_p, std_p = forest.predict_with_uncertainty(data.x[test])
print(f"  mean ensemble standard deviation : {float(std_p.mean())!r}")
print(f"  median                           : {float(np.median(std_p))!r}")
print(f"  maximum                          : {float(std_p.max())!r}")
print(f"  fraction of rows with zero spread: {float(np.mean(std_p == 0.0))!r}")
print()
print("  Is the spread informative? Rows grouped into deciles of ensemble")
print("  standard deviation; if the spread measures anything, the calibration")
print("  gap should grow with it.")
edges = np.quantile(std_p, np.linspace(0.0, 1.0, 11))
edges[0] = -np.inf
edges[-1] = np.inf
which = np.digitize(std_p, edges[1:-1], right=True)
header = (
    f"  {'decile':>7s} {'n':>7s} {'mean std':>10s} {'mean forecast':>14s} "
    f"{'observed':>10s} {'|gap|':>10s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
gaps = []
stds = []
for b in range(10):
    sel = which == b
    if not np.any(sel):
        continue
    mf = float(mean_p[sel].mean())
    ob = float(data.y[test][sel].mean())
    gaps.append(abs(mf - ob))
    stds.append(float(std_p[sel].mean()))
    print(
        f"  {b:>7d} {int(sel.sum()):>7d} {float(std_p[sel].mean()):>10.6f} "
        f"{mf:>14.6f} {ob:>10.6f} {abs(mf - ob):>10.6f}"
    )
rho_s, p_s = spearmanr(stds, gaps)
print(f"  Spearman rank correlation between mean spread and |gap| : {float(rho_s)!r}")
print(f"  p-value (10 deciles)                                    : {float(p_s)!r}")
print("  The relationship is positive and monotone over most of the range, and")
print("  the top decile of spread holds a calibration gap an order of magnitude")
print("  larger than the bottom decile. But with only ten points the rank")
print("  correlation is NOT significant at any conventional level, so the honest")
print("  statement is: the spread is a plausible confidence flag on this record")
print("  and this experiment does not establish it. It is reported as an")
print("  uncertainty output with that caveat, not as a validated one.")

print()
print("[C5] Feature importance of the selected forest (mean impurity decrease)")
imp = forest.feature_importance()
for i in np.argsort(-imp):
    print(f"  {FEATURE_NAMES[i]:<32s} {imp[i]:>10.6f}")
print("  Impurity importance is biased towards high-cardinality features; this")
print("  is an ordering, not an attribution.")

print()
print("[C6] Sensitivity to the horizon")
header = (
    f"  {'horizon [samples]':>18s} {'base rate':>11s} {'constant BS':>13s} "
    f"{'analytic BS':>13s} {'logistic BS':>13s} {'forest BS':>12s} {'winner':>22s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for horizon in (50, 100, 200, 400):
    d = build_outage_dataset(
        series.amplitude,
        threshold=THRESHOLD,
        window_samples=WINDOW,
        horizon_samples=horizon,
        stride_samples=STRIDE,
        gaussian=series.gaussian,
    )
    tr, cl, te = d.split.train, d.split.calibration, d.split.test
    c = ConstantRatePredictor().fit(d.y[tr])
    a = AnalyticLcrPredictor(threshold=THRESHOLD, horizon_samples=horizon).fit(
        series.amplitude[: int(d.index[tr[-1]]) + 1]
    )
    a_cal = PlattCalibrated(a).fit(d.x[cl], d.y[cl])
    lg = LogisticBaseline().fit(d.x[tr], d.y[tr])
    rf = RandomForestOutageClassifier(
        n_estimators=60, min_samples_leaf=best_leaf, seed=SEED
    ).fit(d.x[tr], d.y[tr])
    scores = {
        "constant base rate": brier_score(d.y[te], c.predict_proba_onset(d.x[te])),
        "analytic + Platt": brier_score(d.y[te], a_cal.predict_proba_onset(d.x[te])),
        "logistic regression": brier_score(d.y[te], lg.predict_proba_onset(d.x[te])),
        "random forest": brier_score(d.y[te], rf.predict_proba_onset(d.x[te])),
    }
    win = min(scores, key=lambda k: scores[k])
    print(
        f"  {horizon:>18d} {float(d.y.mean()):>11.5f} "
        f"{scores['constant base rate']:>13.7f} {scores['analytic + Platt']:>13.7f} "
        f"{scores['logistic regression']:>13.7f} {scores['random forest']:>12.7f} "
        f"{win:>22s}"
    )
print("  The forest here uses 60 trees rather than 200 to keep the sweep inside")
print("  the compute budget; the headline table above uses 200.")

print()
print("[C7] Honest limitations of this experiment")
print("  1. One realisation of one synthetic channel. The AR(1) lognormal model")
print("     has the right first-order distribution and an exponential")
print("     autocorrelation, and neither the measured scintillation spectrum nor")
print("     aperture averaging, beam wander or pointing jitter.")
print("  2. The analytic predictor is fitted to the same model family that")
print("     generated the data. On a real record it would be fitting an")
print("     approximation, and its advantage would shrink.")
print("  3. Rows overlap heavily, so the effective sample size is the number of")
print("     independent onsets, a few hundred on the test split. Brier")
print("     differences in the fourth decimal place are not resolvable.")
print("  4. No model here is calibrated across turbulence conditions: the")
print("     channel parameters are stationary over the whole record, which a")
print("     real link's are not over four seconds, let alone four hours.")
print()
print(f"  Compute: 2 cores, 7.8 GiB shared. Longest single fit: the {TREES}-tree")
print(f"  forest on {train.size} rows.")
print()
print("done")
