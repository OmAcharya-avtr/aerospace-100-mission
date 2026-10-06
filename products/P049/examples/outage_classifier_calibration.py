"""Short-horizon outage forecasting: calibration, not accuracy.

Four panels: reliability diagrams for all four forecasters with Wilson
intervals on every bin; the precision-recall curve, which unlike ROC does not
flatter a forecaster on a 2.6 % base rate; the Brier decomposition as stacked
reliability and resolution; and the forest's ensemble spread against the
calibration gap it is supposed to flag.

What to notice: the constant base-rate forecaster sits exactly on the diagonal
and has zero resolution. That is what no skill looks like, and it is the thing
to beat. The raw analytic level-crossing-rate predictor has the highest
discrimination of the four and the *worst* calibration -- it over-forecasts by
more than a factor of two, because turning an expected crossing count into a
probability by Poisson clumping ignores that crossings arrive in bursts. Two
parameters of recalibration fix that and leave it the best forecaster on the
held-out split, ahead of both learned models. A baseline winning is the
published result.

This example uses a 1 500 000-sample record and a 100-tree forest so that it
runs standalone in about a minute; the headline numbers in the README come from
``validation/validate_outage_classifier.py``, which uses 4 000 000 samples and
200 trees.

Writes ``../screenshots/outage_classifier_calibration.png``.

Runtime: about 60 s on two cores.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from linkoutage.calibration import CalibrationReport, evaluate_forecast  # noqa: E402
from linkoutage.channel import lognormal_amplitude_series  # noqa: E402
from linkoutage.features import build_outage_dataset  # noqa: E402
from linkoutage.predictors import (  # noqa: E402
    AnalyticLcrPredictor,
    ConstantRatePredictor,
    LogisticBaseline,
    PlattCalibrated,
    RandomForestOutageClassifier,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "outage_classifier_calibration.png")

FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
N = 1_500_000
SEED = 4901
THRESHOLD = 0.35
WINDOW = 400
HORIZON = 200
STRIDE = 50

series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=SEED)
data = build_outage_dataset(
    series.amplitude,
    threshold=THRESHOLD,
    window_samples=WINDOW,
    horizon_samples=HORIZON,
    stride_samples=STRIDE,
    gaussian=series.gaussian,
)
train, cal, test = data.split.train, data.split.calibration, data.split.test
print(data.report())

# Baselines first.
constant = ConstantRatePredictor().fit(data.y[train])
analytic = AnalyticLcrPredictor(threshold=THRESHOLD, horizon_samples=HORIZON).fit(
    series.amplitude[: int(data.index[train[-1]]) + 1]
)
logistic = LogisticBaseline().fit(data.x[train], data.y[train])
# Then the learned model.
forest = RandomForestOutageClassifier(
    n_estimators=100, min_samples_leaf=5, seed=SEED
).fit(data.x[train], data.y[train])
analytic_platt = PlattCalibrated(analytic).fit(data.x[cal], data.y[cal])

models = (constant, analytic, logistic, forest, analytic_platt)
colours = ("#444444", "#c00000", "#1f4e79", "#2e7d32", "#7b1fa2")
reports = [
    evaluate_forecast(
        data.y[test],
        m.predict_proba_onset(data.x[test]),
        name=m.name,
        reference_rate=constant.rate,
    )
    for m in models
]

print()
print(CalibrationReport.header())
for report in reports:
    print(report.line())
best = min(reports, key=lambda r: r.brier)
print()
print(f"lowest Brier on the held-out split: {best.name}")
print(f"all-negative accuracy on the same split: {reports[0].accuracy_all_negative:.6f}")

figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.6))

# --- panel 1: reliability ----------------------------------------------------
ax = axes[0, 0]
upper = max(float(r.curve.mean_forecast.max()) for r in reports) * 1.1
ax.plot([0, upper], [0, upper], lw=1.0, ls="--", color="0.5", label="perfect calibration")
for report, colour in zip(reports, colours, strict=True):
    curve = report.curve
    ax.errorbar(
        curve.mean_forecast,
        curve.observed_frequency,
        yerr=[
            curve.observed_frequency - curve.lower,
            curve.upper - curve.observed_frequency,
        ],
        fmt="o-",
        ms=4,
        lw=1.2,
        capsize=2,
        color=colour,
        label=f"{report.name} (ECE {report.ece:.4f})",
    )
ax.axhline(
    float(data.y[test].mean()), lw=0.9, ls=":", color="0.3",
)
ax.set_xlim(0, upper)
ax.set_ylim(0, upper)
ax.set_xlabel("mean forecast probability")
ax.set_ylabel("observed frequency")
ax.set_title(
    "Reliability, ten equal-count bins, Wilson 95 % intervals", fontsize=10
)
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.25)

# --- panel 2: precision-recall ----------------------------------------------
ax = axes[0, 1]
# The constant forecaster is omitted here: with one distinct score its
# precision-recall curve is the degenerate interpolation between (0, 1) and
# (1, base rate), which looks like a good curve and is not one. Its average
# precision, which equals the base rate, is in the legend of panel 1 instead.
for model, report, colour in zip(models, reports, colours, strict=True):
    if isinstance(model, ConstantRatePredictor):
        continue
    p = model.predict_proba_onset(data.x[test])
    precision, recall, _ = precision_recall_curve(data.y[test], p)
    ax.plot(
        recall,
        precision,
        lw=1.4,
        color=colour,
        label=f"{report.name} (AP {report.average_precision:.3f})",
    )
base = float(data.y[test].mean())
ax.axhline(base, lw=1.0, ls="--", color="0.5", label=f"base rate {base:.4f}")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title("Precision-recall: the honest curve at a 2.6 % base rate", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)

# --- panel 3: Brier decomposition -------------------------------------------
ax = axes[1, 0]
names = ["constant", "analytic", "logistic", "forest", "analytic\n+ Platt"]
x = np.arange(len(reports))
rel = np.array([r.reliability for r in reports])
res = np.array([r.resolution for r in reports])
ax.bar(x - 0.2, rel, width=0.38, color="#c00000", label="reliability (lower better)")
ax.bar(x + 0.2, res, width=0.38, color="#2e7d32", label="resolution (higher better)")
ax.set_yscale("symlog", linthresh=1e-6)
ax.set_ylim(0.0, max(rel.max(), res.max()) * 12.0)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("contribution to the Brier score")
ax.set_title(
    "BS = reliability - resolution + uncertainty"
    f"  (uncertainty = {reports[0].uncertainty:.5f})",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
ax.annotate(
    "no resolution bar here:\nthe constant forecaster\nhas zero skill, exactly",
    xy=(0.2, 2e-6),
    xytext=(0.45, 2e-4),
    arrowprops={"arrowstyle": "->", "lw": 0.9},
    fontsize=8,
)

# --- panel 4: ensemble spread as a confidence flag ---------------------------
ax = axes[1, 1]
mean_p, std_p = forest.predict_with_uncertainty(data.x[test])
edges = np.quantile(std_p, np.linspace(0.0, 1.0, 11))
edges[0] = -np.inf
edges[-1] = np.inf
which = np.digitize(std_p, edges[1:-1], right=True)
mean_std = []
gap = []
for b in range(10):
    sel = which == b
    if not np.any(sel):
        continue
    mean_std.append(float(std_p[sel].mean()))
    gap.append(abs(float(mean_p[sel].mean()) - float(data.y[test][sel].mean())))
ax.loglog(mean_std, gap, "o-", lw=1.4, ms=5, color="#2e7d32")
ax.set_ylim(min(gap) * 0.6, max(gap) * 2.5)
ax.set_xlabel("mean ensemble standard deviation in the decile")
ax.set_ylabel("|mean forecast - observed frequency|")
ax.set_title(
    "Is the forest's spread a usable confidence flag?\n"
    "Positive but not established on ten deciles",
    fontsize=10,
)
ax.grid(alpha=0.25, which="both")
ax.annotate(
    f"max spread {float(std_p.max()):.3f}\n"
    f"median spread {float(np.median(std_p)):.3f}",
    xy=(mean_std[-1], gap[-1]),
    xytext=(mean_std[0] * 1.6, max(gap) * 0.25),
    arrowprops={"arrowstyle": "->", "lw": 0.9},
    fontsize=8,
)

figure.suptitle(
    "Short-horizon outage forecasting on a correlated lognormal link\n"
    f"threshold {THRESHOLD}, horizon {HORIZON} samples, base rate "
    f"{data.base_rate:.4f}, {int(data.y[test].sum())} test onsets; "
    f"lowest Brier: {best.name}",
    fontsize=11,
)
figure.subplots_adjust(
    left=0.055, right=0.985, top=0.905, bottom=0.065, wspace=0.20, hspace=0.26
)
figure.savefig(OUT, dpi=130)
print()
print(f"wrote screenshots/{os.path.basename(OUT)}")
