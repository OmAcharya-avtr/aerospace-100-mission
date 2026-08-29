"""Validation script 2: learned model vs classical baselines on held-out data,
plus prediction-interval coverage.

Run from the product root:

    python validation/benchmark_ml.py | tee validation/benchmark_results.md

Every number in ``VALIDATION.md`` (learned-vs-baseline comparison and
prediction-interval coverage sections) comes from this script. Nothing here
is asserted without being computed in this run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope.model import (  # noqa: E402
    DimmOnlyBaseline,
    MeanTrainingBaseline,
    ScintillometerWeakBaseline,
    interval_coverage,
    train_default_model,
)
from turbscope.scintillometer import is_weak_regime, rytov_variance  # noqa: E402
from turbscope.synthetic import SCINT_WAVELENGTH_M, WAVE_TYPE  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def errors_dex(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "p95_abs": float(np.percentile(np.abs(err), 95)),
    }


def main() -> int:
    t0 = time.time()
    print("TurbScope learned-model benchmark -- all values computed in this run")
    print(f"numpy {np.__version__}")

    # ------------------------------------------------------------------ fit
    model, art = train_default_model()
    model_raw, art_raw = train_default_model(calibrate=False)
    fit_time = time.time() - t0
    print(
        f"train_default_model(): fit+calibration wall time {fit_time:.2f} s "
        f"({len(art['fit_scenarios'])} fit / {len(art['calibration_scenarios'])} calibration / "
        f"{len(art['test_scenarios'])} test scenarios, x_test shape {art['x_test'].shape})"
    )
    print(f"fit_report: {model.fit_report()}")
    print(f"conformal delta: {model.conformal_delta_dex:.6f} dex")

    x_test, y_test = art["x_test"], art["y_test"]

    # ------------------------------------------------------ true regime tags
    # Recompute the TRUE Rytov variance for every test row from its true Cn2
    # (y_test) and known path length (feature 3), at the fixed scintillometer
    # wavelength/wave-type, to tag each row weak/saturated for the breakdown.
    cn2_true = 10.0**y_test
    path_length = 10.0 ** x_test[:, 3]
    # rytov_variance takes a scalar path length; the test rows have varying
    # path lengths, so it is evaluated row-by-row (675 rows, negligible cost).
    r_var_true = np.array(
        [
            float(rytov_variance(c, length_i, SCINT_WAVELENGTH_M, WAVE_TYPE))
            for c, length_i in zip(cn2_true, path_length, strict=True)
        ]
    )
    weak_mask = np.asarray(is_weak_regime(r_var_true))

    # --------------------------------------------------------------- fit
    scint_base = ScintillometerWeakBaseline()
    dimm_base = DimmOnlyBaseline()
    mean_base = MeanTrainingBaseline().fit(art["x_train"], art["y_train"])

    predictors = {
        "TurbScope learned model": model.predict_log10_cn2(x_test),
        "Scintillometer weak baseline (mandated)": scint_base.predict_log10_cn2(x_test),
        "DIMM-only baseline": dimm_base.predict_log10_cn2(x_test),
        "Training mean (learned-nothing floor)": mean_base.predict_log10_cn2(x_test),
    }

    # ---------------------------------------------------------------- 1
    section("1. Overall held-out error, all test rows (dex = decades of Cn2)")
    print(f"{'predictor':<42}{'RMSE':>10}{'MAE':>10}{'bias':>10}{'p95':>10}")
    rows_overall = {}
    for name, pred in predictors.items():
        e = errors_dex(y_test, pred)
        rows_overall[name] = e
        print(f"{name:<42}{e['rmse']:>10.4f}{e['mae']:>10.4f}{e['bias']:>10.4f}{e['p95_abs']:>10.4f}")

    ratio = rows_overall["TurbScope learned model"]["rmse"] / rows_overall[
        "Scintillometer weak baseline (mandated)"
    ]["rmse"]
    print(f"\nlearned/baseline RMSE ratio (mandated comparison): {ratio:.4f}")

    # ---------------------------------------------------------------- 2
    section("2. Error broken down by TRUE regime (weak vs saturated scintillometer path)")
    print(f"n weak rows: {int(np.sum(weak_mask))}   n saturated rows: {int(np.sum(~weak_mask))}")
    print(f"{'predictor':<42}{'RMSE weak':>12}{'RMSE saturated':>16}")
    for name, pred in predictors.items():
        e_weak = errors_dex(y_test[weak_mask], pred[weak_mask])
        e_sat = errors_dex(y_test[~weak_mask], pred[~weak_mask])
        print(f"{name:<42}{e_weak['rmse']:>12.4f}{e_sat['rmse']:>16.4f}")

    # ---------------------------------------------------------------- 3
    section("3. Prediction-interval coverage on held-out test data (nominal 90%)")
    lo_raw, mid_raw, hi_raw = model_raw._three(x_test)  # noqa: SLF001 -- validation script
    lo_cal, mid_cal, hi_cal = model._three(x_test)  # noqa: SLF001

    cov_raw, width_raw = interval_coverage(y_test, lo_raw, hi_raw)
    cov_cal, width_cal = interval_coverage(y_test, lo_cal, hi_cal)
    print(f"{'interval':<30}{'nominal':>10}{'coverage':>12}{'mean width (dex)':>20}")
    print(f"{'raw quantile GBR':<30}{model.coverage:>10.3f}{cov_raw:>12.4f}{width_raw:>20.4f}")
    cal_row = f"{'conformally calibrated':<30}{model.coverage:>10.3f}{cov_cal:>12.4f}"
    print(cal_row + f"{width_cal:>20.4f}")

    section("3b. Coverage by regime (calibrated model)")
    cov_weak, width_weak = interval_coverage(
        y_test[weak_mask], lo_cal[weak_mask], hi_cal[weak_mask]
    )
    cov_sat, width_sat = interval_coverage(
        y_test[~weak_mask], lo_cal[~weak_mask], hi_cal[~weak_mask]
    )
    print(f"weak regime      : coverage {cov_weak:.4f}, mean width {width_weak:.4f} dex")
    print(f"saturated regime : coverage {cov_sat:.4f}, mean width {width_sat:.4f} dex")

    # ---------------------------------------------------------------- 4
    section("4. Reproducibility check")
    model2, art2 = train_default_model()
    pred1 = model.predict_log10_cn2(x_test)
    pred2 = model2.predict_log10_cn2(art2["x_test"])
    diff = np.max(np.abs(pred1 - pred2))
    same_x = bool(np.array_equal(x_test, art2["x_test"]))
    print(f"identical test features on re-run: {same_x}")
    print(f"max |prediction difference| across re-runs: {diff:.3e} dex")
    print(f"conformal delta identical: {model.conformal_delta_dex == model2.conformal_delta_dex}")

    total_time = time.time() - t0
    section("Summary")
    print(f"total script wall time: {total_time:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
