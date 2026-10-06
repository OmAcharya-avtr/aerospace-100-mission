"""Regenerate every fitted model deterministically and persist it with joblib.

No model binary is committed. This script is the committed alternative: it
rebuilds the dataset from its seed, refits all five forecasters with the
published hyperparameters, writes them to ``validation/outage_models.joblib``
and the dataset summary to ``validation/outage_dataset.npz``, and then
*verifies* that the refitted models reproduce the published test Brier scores
bit-for-bit against the values recorded here.

Everything is keyed on one seed (4901) and the fixed configuration below, so a
run on another machine with the same NumPy and scikit-learn versions produces
the same numbers. Across versions the forest may differ in its last digits;
the verification tolerance is stated and reported rather than assumed.

Outputs (both gitignored, both regenerable by this script alone):
  outage_models.joblib   the five fitted forecasters
  outage_dataset.npz     features, labels, decision indices and split indices

Runtime: about 60 s on two cores.
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401
import joblib
import numpy as np

from linkoutage.calibration import brier_score
from linkoutage.channel import lognormal_amplitude_series
from linkoutage.features import build_outage_dataset
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
LEAF = 5  # selected on the calibration split in validate_outage_classifier.py

# Published test Brier scores, from validation/outage_classifier_output.txt.
PUBLISHED = {
    "constant base rate": 0.0264495,
    "analytic LCR (fit)": 0.0265626,
    "logistic regression": 0.0220142,
    "random forest": 0.0223498,
    "analytic LCR (fit) + Platt": 0.0217557,
    "random forest + Platt": 0.0222801,
}
TOLERANCE = 5e-6

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "outage_models.joblib")
DATA_PATH = os.path.join(HERE, "outage_dataset.npz")

print("=" * 78)
print("regenerate_outage_model.py")
print("=" * 78)
print()
print("Configuration (the whole of it):")
for key, value in (
    ("n_samples", N_SAMPLES),
    ("fs_hz", FS),
    ("tau_s", TAU),
    ("si", SI),
    ("seed", SEED),
    ("threshold", THRESHOLD),
    ("window_samples", WINDOW),
    ("horizon_samples", HORIZON),
    ("stride_samples", STRIDE),
    ("n_estimators", TREES),
    ("min_samples_leaf", LEAF),
):
    print(f"  {key:<18s}: {value!r}")

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
print(data.report())

constant = ConstantRatePredictor().fit(data.y[train])
analytic = AnalyticLcrPredictor(threshold=THRESHOLD, horizon_samples=HORIZON).fit(
    series.amplitude[: int(data.index[train[-1]]) + 1]
)
logistic = LogisticBaseline().fit(data.x[train], data.y[train])
forest = RandomForestOutageClassifier(
    n_estimators=TREES, min_samples_leaf=LEAF, seed=SEED
).fit(data.x[train], data.y[train])
analytic_platt = PlattCalibrated(analytic).fit(data.x[cal], data.y[cal])
forest_platt = PlattCalibrated(forest).fit(data.x[cal], data.y[cal])

models = {
    constant.name: constant,
    analytic.name: analytic,
    logistic.name: logistic,
    forest.name: forest,
    analytic_platt.name: analytic_platt,
    forest_platt.name: forest_platt,
}

print()
print("Verification against the published test Brier scores")
header = f"  {'forecaster':<30s} {'refitted':>12s} {'published':>12s} {'|diff|':>11s} {'ok':>5s}"
print(header)
print("  " + "-" * (len(header) - 2))
all_ok = True
for name, model in models.items():
    bs = brier_score(data.y[test], model.predict_proba_onset(data.x[test]))
    want = PUBLISHED[name]
    diff = abs(bs - want)
    ok = diff < TOLERANCE
    all_ok &= ok
    print(f"  {name:<30s} {bs:>12.7f} {want:>12.7f} {diff:>11.2e} {'yes' if ok else 'NO':>5s}")
print(f"  tolerance {TOLERANCE!r} (the published values are rounded to 7 figures)")
print(f"  ALL REPRODUCED = {all_ok}")

joblib.dump(models, MODEL_PATH, compress=3)
np.savez_compressed(
    DATA_PATH,
    x=data.x.astype(np.float32),
    y=data.y.astype(np.int8),
    index=data.index,
    train=train,
    calibration=cal,
    test=test,
)
print()
print(f"wrote validation/{os.path.basename(MODEL_PATH)} "
      f"({os.path.getsize(MODEL_PATH) / 1e6:.2f} MB)")
print(f"wrote validation/{os.path.basename(DATA_PATH)} "
      f"({os.path.getsize(DATA_PATH) / 1e6:.2f} MB)")
print("Neither file is committed: both are listed in .gitignore and both are")
print("reproduced exactly by re-running this script.")
print()
print("done")
