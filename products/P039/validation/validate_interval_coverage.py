#!/usr/bin/env python3
"""Validation 2 -- prediction intervals against their nominal coverage.

What is being checked
---------------------
Every model in this package reports an interval, not a point. An interval is
worth nothing unless the fraction of held-out pipelines it actually contains
matches the fraction it claims. This script measures that fraction on pipelines
no model has seen, at three nominal levels, in both dependence regimes, for
both native and split-conformal intervals, and reports every measured coverage
next to its nominal value with the binomial standard error

    SE = sqrt(c (1 - c) / n)

so the reader can tell a real mis-calibration from sampling noise. With
n = 150 held-out pipelines and a nominal 0.90, SE = 0.024, which means a
measured 0.86 is within two standard errors of nominal and a measured 0.65 is
not.

What is expected to fail, and does
----------------------------------
The analytic model's native interval propagates probe sampling uncertainty
only (JCGM 100:2008). It contains no term for Fenton-Wilkinson approximation
error and, in independence mode, no term for stage dependence. It is therefore
predicted in advance to UNDER-cover, badly in the correlated regime. That is
reported as a failure against nominal, not excused: an interval that claims
90 % and delivers much less is a defect in the interval, and the honest fix is
the conformal wrapper, whose coverage is also measured here.

Reproduce with:
    cd validation && PYTHONPATH=../src python3 validate_interval_coverage.py

Runtime: about 20 s on one uncontended core.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from latencynet.conformal import ConformalPredictor, coverage_quantisation
from latencynet.dataset import build_dataset, log_target
from latencynet.learned import LearnedTailPredictor
from latencynet.linear import LinearTailPredictor
from latencynet.metrics import interval_coverage
from latencynet.predictors import AnalyticTailPredictor

N_TRAIN = 200
N_CALIBRATION = 60
N_TEST = 150
N_PROBE = 256
N_REFERENCE = 25_000
N_REFERENCE_TEST = 90_000
SEED = 20260402
LEVELS = (0.5, 0.8, 0.9)
#: The level the learned model's quantile heads are fitted for. Refitting
#: three 300-tree ensembles per level costs more runtime than this build has,
#: so the learned model's NATIVE interval is reported at this level only; its
#: conformal interval, which needs no refit, is reported at every level.
LEARNED_NATIVE_LEVEL = 0.9
PROBABILITIES = (0.99, 0.999)
OUTPUT_JSON = Path(__file__).resolve().parent / "interval_coverage.json"

#: A measured coverage is reported as consistent with nominal when it sits
#: within this many binomial standard errors of it. Two standard errors is a
#: 95 % two-sided band and is stated here before any number was looked at.
SIGMA_BAND = 2.0


def main() -> int:
    print("=" * 94)
    print("P039 latencynet -- Validation 2: prediction-interval coverage on held-out pipelines")
    print("=" * 94)
    print(
        f"splits: train={N_TRAIN} calibration={N_CALIBRATION} test={N_TEST} pipelines; "
        f"probe n={N_PROBE}"
    )
    print(
        f"reference sample: {N_REFERENCE} passes for train/calibration, "
        f"{N_REFERENCE_TEST} for test"
    )
    print(
        f"binomial SE at nominal 0.90 with n={N_TEST}: "
        f"{np.sqrt(0.9 * 0.1 / N_TEST):.4f}; conformal coverage is quantised in steps of "
        f"1/(m+1) = {coverage_quantisation(N_CALIBRATION):.4f}"
    )
    print("No measured wall-clock time appears anywhere in this script.")

    payload: dict[str, object] = {
        "splits": {"train": N_TRAIN, "calibration": N_CALIBRATION, "test": N_TEST},
        "n_probe": N_PROBE,
        "n_reference": N_REFERENCE,
        "n_reference_test": N_REFERENCE_TEST,
        "seed": SEED,
        "sigma_band": SIGMA_BAND,
        "rows": [],
    }
    rows: list[dict[str, object]] = []
    verdicts: list[tuple[str, bool]] = []

    for regime in ("independent", "correlated"):
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
        for p in PROBABILITIES:
            truth = log_target(dataset.test, p)
            print(f"\nregime={regime}  target=p{p * 100:g}")
            print("-" * 94)
            print(
                f"  {'model':<26}{'interval':<11}{'nominal':>9}{'measured':>10}"
                f"{'SE':>8}{'dev/SE':>9}{'width':>9}{'verdict':>12}"
            )
            # Fitted once per (regime, p). The analytic and OLS models are
            # level-independent once fitted; only their interval is
            # recomputed per level. The learned model's quantile heads are
            # level-specific, so its native interval is reported only at
            # LEARNED_NATIVE_LEVEL.
            models = [
                AnalyticTailPredictor(assume_independent=True).fit(dataset.train, p),
                LinearTailPredictor().fit(dataset.train, p),
                LearnedTailPredictor(native_interval_level=LEARNED_NATIVE_LEVEL).fit(
                    dataset.train, p
                ),
                AnalyticTailPredictor(assume_independent=False).fit(dataset.train, p),
            ]
            for level in LEVELS:
                for model in models:
                    kinds: list[tuple[str, object]] = []
                    if not isinstance(model, LearnedTailPredictor) or level == LEARNED_NATIVE_LEVEL:
                        kinds.append(
                            (
                                "native",
                                interval_coverage(
                                    model.predict_log_interval(dataset.test, level=level), truth
                                ),
                            )
                        )
                    wrapper = ConformalPredictor.from_fitted(model, p, level=level)
                    wrapper.calibrate(dataset.calibration)
                    kinds.append(
                        (
                            "conformal",
                            interval_coverage(
                                wrapper.predict_log_interval(dataset.test), truth
                            ),
                        )
                    )
                    for kind, cov in kinds:
                        ok = abs(cov.deviation_sigma) <= SIGMA_BAND
                        verdicts.append((f"{regime}/p{p}/{model.name}/{kind}/L{level}", ok))
                        print(
                            f"  {model.name:<26}{kind:<11}{cov.nominal:>9.2f}"
                            f"{cov.measured:>10.3f}{cov.standard_error:>8.3f}"
                            f"{cov.deviation_sigma:>9.2f}{cov.mean_log_width:>9.4f}"
                            f"{'PASS' if ok else 'FAIL':>12}"
                        )
                        rows.append(
                            {
                                "regime": regime,
                                "p": p,
                                "model": model.name,
                                "interval": kind,
                                "nominal": cov.nominal,
                                "measured": cov.measured,
                                "standard_error": cov.standard_error,
                                "deviation_sigma": cov.deviation_sigma,
                                "wilson_lower": cov.wilson_lower,
                                "wilson_upper": cov.wilson_upper,
                                "n": cov.n,
                                "n_covered": cov.n_covered,
                                "mean_log_width": cov.mean_log_width,
                                "within_band": bool(ok),
                                "one_sided_guarantee_held": bool(
                                    cov.measured
                                    >= cov.nominal - SIGMA_BAND * cov.standard_error
                                ),
                            }
                        )

    payload["rows"] = rows
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    native_rows = [r for r in rows if r["interval"] == "native"]
    conf_rows = [r for r in rows if r["interval"] == "conformal"]
    print("\n" + "=" * 94)
    print("summary")
    print("-" * 94)
    for label, group in (("native", native_rows), ("conformal", conf_rows)):
        ok = sum(1 for r in group if r["within_band"])
        print(f"  {label:<11}{ok}/{len(group)} rows within {SIGMA_BAND:g} binomial SE of nominal")
    for model in ("analytic_sum_indep", "linear_ols", "learned_gbt", "analytic_sum_cov"):
        nat = [r for r in native_rows if r["model"] == model]
        con = [r for r in conf_rows if r["model"] == model]
        print(
            f"  {model:<26}native {sum(1 for r in nat if r['within_band'])}/{len(nat)}   "
            f"conformal {sum(1 for r in con if r['within_band'])}/{len(con)}   "
            f"mean conformal width {np.mean([float(r['mean_log_width']) for r in con]):.4f}"
        )
    held = sum(1 for r in conf_rows if r["one_sided_guarantee_held"])
    over = sum(
        1
        for r in conf_rows
        if not r["within_band"] and float(r["measured"]) > float(r["nominal"])
    )
    print(
        f"  conformal one-sided guarantee (measured >= nominal - "
        f"{SIGMA_BAND:g} SE) held on {held}/{len(conf_rows)} rows"
    )
    print(
        f"  of the {len(conf_rows) - sum(1 for r in conf_rows if r['within_band'])} conformal "
        f"rows outside the two-sided band, {over} are OVER-coverage, which is the direction the "
        f"split-conformal\n  guarantee permits: its coverage lies in "
        f"[level, level + 1/(m+1)] in expectation, i.e. up to "
        f"{coverage_quantisation(N_CALIBRATION):.4f} above nominal, and the two-sided band does "
        f"not allow for that offset."
    )
    print(
        "\n  Reading: the conformal interval is the one to use. The analytic\n"
        "  model's native interval under-covers because it propagates probe\n"
        "  sampling uncertainty and nothing else; that is a property of the\n"
        "  interval, stated in latencynet.analytic before it was measured,\n"
        "  and it is reported as FAIL rather than re-banded."
    )
    print(f"\n  written: {OUTPUT_JSON.name}")
    n_ok = sum(1 for _, ok in verdicts if ok)
    print("=" * 94)
    print(f"coverage rows within band: {n_ok}/{len(verdicts)}")
    print(
        "exit status 0: this script reports coverage, it does not gate on it. "
        "Failures above are findings."
    )
    print("=" * 94)
    return 0


if __name__ == "__main__":
    sys.exit(main())
