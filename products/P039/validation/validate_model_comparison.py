#!/usr/bin/env python3
"""Validation 4 -- the three-way comparison, and which case the data falls in.

The question the specification asks
-----------------------------------
"The analytic baseline is expected to win where stages are independent; the
learned model should only help where they are not, and the README must show
which case the data falls in." This script answers that, in both regimes, at
both tail probabilities, on held-out pipelines, and the answer it produces is
what the README reports -- including where it contradicts the expectation.

Protocol, fixed in latencynet.compare before the run
----------------------------------------------------
Split by pipeline into train / calibration / test. Fit baseline 1 (analytic
sum-of-stages, nothing to fit), baseline 2 (OLS on the thirteen probe
features) and the learned model (gradient-boosted trees) on train only.
Calibrate a split-conformal wrapper for each on the calibration split. Score
on test. The winner is the model with the smallest mean absolute log error,
and a win is called significant only at ``p < 0.05`` on a paired t-test
against the analytic baseline, over the same held-out pipelines.

A covariance-aware analytic variant is scored alongside as a diagnostic. It is
not one of the specified baselines; it is there so that any learned-model
advantage can be separated from the much simpler statement "measure the
covariance and put it in equation (2)".

Capacity sweep
--------------
Section (c) refits the learned model at six ``(n_estimators, max_depth,
learning_rate)`` settings spanning a factor of sixteen in ensemble size, so
that the comparison's conclusion can be checked against the possibility that
it is an artefact of one hyperparameter choice. The sweep is reported in full,
every row, and is not reduced to its best entry.

Reproduce with:
    cd validation && PYTHONPATH=../src python3 validate_model_comparison.py

Runtime: about 35 s on one uncontended core.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from latencynet.compare import compare_models, format_comparison
from latencynet.dataset import build_dataset, log_target
from latencynet.features import FEATURE_NAMES
from latencynet.learned import BoostingHyperparameters, LearnedTailPredictor
from latencynet.linear import LinearTailPredictor
from latencynet.metrics import log_accuracy
from latencynet.predictors import AnalyticTailPredictor

N_TRAIN = 200
N_CALIBRATION = 60
N_TEST = 150
N_PROBE = 256
N_REFERENCE = 25_000
N_REFERENCE_TEST = 90_000
SEED = 20260402
LEVEL = 0.9
PROBABILITIES = (0.99, 0.999)
SWEEP_PROBABILITY = 0.99
SWEEP = (
    (50, 2, 0.05),
    (100, 3, 0.05),
    (300, 3, 0.05),
    (300, 2, 0.10),
    (300, 5, 0.05),
    (800, 3, 0.02),
)
OUTPUT_JSON = Path(__file__).resolve().parent / "model_comparison.json"


def capacity_sweep(dataset, regime: str) -> tuple[list[dict[str, object]], LearnedTailPredictor]:
    print(f"\n(c) learned-model capacity sweep, regime={regime}, p={SWEEP_PROBABILITY}")
    print("-" * 102)
    truth = log_target(dataset.test, SWEEP_PROBABILITY)
    analytic = AnalyticTailPredictor().fit(dataset.train, SWEEP_PROBABILITY)
    linear = LinearTailPredictor().fit(dataset.train, SWEEP_PROBABILITY)
    base_a = log_accuracy(analytic.predict_log(dataset.test), truth).mean_abs_log_error
    base_l = log_accuracy(linear.predict_log(dataset.test), truth).mean_abs_log_error
    print(f"  reference: analytic_sum_indep {base_a:.5f}   linear_ols {base_l:.5f}")
    print(f"  {'n_estimators':>13}{'max_depth':>11}{'learning_rate':>15}{'mean|dlnq|':>12}"
          f"{'beats analytic':>16}{'beats linear':>14}")
    rows: list[dict[str, object]] = []
    reference_model: LearnedTailPredictor | None = None
    for n_est, depth, lr in SWEEP:
        model = LearnedTailPredictor(
            BoostingHyperparameters(n_estimators=n_est, max_depth=depth, learning_rate=lr)
        ).fit(dataset.train, SWEEP_PROBABILITY)
        err = log_accuracy(model.predict_log(dataset.test), truth).mean_abs_log_error
        if (n_est, depth, lr) == (300, 3, 0.05):
            reference_model = model
        print(
            f"  {n_est:>13}{depth:>11}{lr:>15.2f}{err:>12.5f}"
            f"{'yes' if err < base_a else 'no':>16}{'yes' if err < base_l else 'no':>14}"
        )
        rows.append(
            {
                "regime": regime,
                "p": SWEEP_PROBABILITY,
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "mean_abs_log_error": err,
                "beats_analytic": bool(err < base_a),
                "beats_linear": bool(err < base_l),
            }
        )
    if reference_model is None:  # pragma: no cover - SWEEP contains the default
        raise RuntimeError("the sweep must include the default hyperparameters")
    return rows, reference_model


def main() -> int:
    print("=" * 102)
    print("P039 latencynet -- Validation 4: three-way held-out comparison in both regimes")
    print("=" * 102)
    print(
        f"splits: train={N_TRAIN} calibration={N_CALIBRATION} test={N_TEST} pipelines; "
        f"probe n={N_PROBE}; nominal level={LEVEL}"
    )
    print(
        f"reference sample: {N_REFERENCE} passes for train/calibration, "
        f"{N_REFERENCE_TEST} for test; seed {SEED}"
    )
    print(
        "winner rule, fixed in advance: smallest mean absolute log error on the test split; "
        "significance by\npaired t-test against analytic_sum_indep at p < 0.05."
    )
    print("No measured wall-clock time appears anywhere in this script.")

    payload: dict[str, object] = {
        "splits": {"train": N_TRAIN, "calibration": N_CALIBRATION, "test": N_TEST},
        "n_probe": N_PROBE,
        "n_reference": N_REFERENCE,
        "n_reference_test": N_REFERENCE_TEST,
        "seed": SEED,
        "level": LEVEL,
        "winner_rule": "smallest mean absolute log error on the held-out split",
        "comparisons": [],
        "sweep": [],
        "feature_importance": [],
    }
    comparisons: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    winners: list[tuple[str, float, str, bool]] = []

    for section, regime in (("a", "independent"), ("b", "correlated")):
        dataset = build_dataset(
            regime,
            n_train=N_TRAIN,
            n_calibration=N_CALIBRATION,
            n_test=N_TEST,
            seed=SEED,
            n_probe=N_PROBE,
            n_reference=N_REFERENCE,
            n_reference_test=N_REFERENCE_TEST,
        )
        rhos = [r.latent_rho for r in dataset.all_records()]
        print(f"\n({section}) regime={regime}")
        print("-" * 102)
        print(
            f"  injected latent correlation across {len(rhos)} pipelines: "
            f"min {min(rhos):.3f} mean {float(np.mean(rhos)):.3f} max {max(rhos):.3f}"
        )
        stage_counts = np.array([r.n_stages for r in dataset.all_records()])
        print(
            f"  stages per pipeline: min {stage_counts.min()} mean "
            f"{stage_counts.mean():.2f} max {stage_counts.max()}"
        )
        for p in PROBABILITIES:
            result = compare_models(dataset, p, level=LEVEL)
            print()
            print(format_comparison(result))
            winners.append((f"{regime}/p{p}", p, result.winner, result.winner_is_significant))
            comparisons.append(
                {
                    "regime": regime,
                    "p": p,
                    "winner": result.winner,
                    "winner_is_significant": result.winner_is_significant,
                    "mean_reference_relative_se": result.mean_reference_relative_se,
                    "coverage_quantisation": result.coverage_quantisation,
                    "models": [
                        {
                            "name": s.name,
                            "is_baseline": s.is_baseline,
                            "mean_abs_log_error": s.accuracy.mean_abs_log_error,
                            "median_abs_log_error": s.accuracy.median_abs_log_error,
                            "rms_log_error": s.accuracy.rms_log_error,
                            "bias_log": s.accuracy.bias_log,
                            "p90_abs_log_error": s.accuracy.p90_abs_log_error,
                            "mean_relative_error": s.accuracy.mean_relative_error,
                            "native_coverage": (
                                None if s.native_coverage is None else s.native_coverage.measured
                            ),
                            "native_coverage_se": (
                                None
                                if s.native_coverage is None
                                else s.native_coverage.standard_error
                            ),
                            "conformal_coverage": s.conformal_coverage.measured,
                            "conformal_coverage_se": s.conformal_coverage.standard_error,
                            "conformal_mean_log_width": s.conformal_coverage.mean_log_width,
                            "paired_mean_difference": s.paired_mean_difference,
                            "paired_p_value": s.paired_p_value,
                        }
                        for s in result.scores
                    ],
                }
            )
        regime_rows, learned = capacity_sweep(dataset, regime)
        sweep_rows += regime_rows
        imp = learned.feature_importances()
        order = np.argsort(imp)[::-1]
        print(f"\n  learned-model impurity feature importances, regime={regime}, "
              f"p={SWEEP_PROBABILITY} (top 6)")
        for i in order[:6]:
            print(f"    {FEATURE_NAMES[i]:<24}{imp[i]:.4f}")
        print(
            "    impurity importances are biased toward high-cardinality features\n"
            "    (Strobl et al. 2007); reported as what the model leaned on, not as a\n"
            "    causal claim."
        )
        importance_rows.append(
            {
                "regime": regime,
                "p": SWEEP_PROBABILITY,
                "importances": {FEATURE_NAMES[i]: float(imp[i]) for i in range(len(imp))},
            }
        )

    payload["comparisons"] = comparisons
    payload["sweep"] = sweep_rows
    payload["feature_importance"] = importance_rows
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + "=" * 102)
    print("which case does the data fall in")
    print("-" * 102)
    for key, _p, winner, sig in winners:
        print(f"  {key:<26}winner {winner:<22}"
              f"{'significant vs analytic baseline' if sig else 'not significant vs baseline'}")
    beats_linear = sum(1 for r in sweep_rows if r["beats_linear"])
    beats_analytic = sum(1 for r in sweep_rows if r["beats_analytic"])
    print(
        f"\n  capacity sweep: the learned model beat the analytic baseline on "
        f"{beats_analytic}/{len(sweep_rows)} settings\n"
        f"  and the linear baseline on {beats_linear}/{len(sweep_rows)} settings."
    )
    print(f"\n  written: {OUTPUT_JSON.name}")
    print("=" * 102)
    return 0


if __name__ == "__main__":
    sys.exit(main())
