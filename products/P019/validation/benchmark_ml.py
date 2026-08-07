"""Validation script 2: learned model vs the published baselines, held-out data.

Run from the product root:

    python validation/benchmark_ml.py | tee validation/benchmark_results.md

Produces §4-§6 of ``VALIDATION.md``: the error table against HV 5/7 and the
other comparators, the interval-coverage table (raw quantile models and
conformally calibrated), the seeing quantities of a predicted profile with the
raw numbers needed for the hand check, and a reproducibility check.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sklearn  # noqa: E402

from cncast.baselines import bufton_wind  # noqa: E402
from cncast.dataset import default_altitude_grid, profile_cn2  # noqa: E402
from cncast.model import (  # noqa: E402
    ClimatologyBaseline,
    CnCastModel,
    Hv57Baseline,
    SlcBaseline,
    interval_coverage,
    train_default_model,
)
from cncast.seeing import (  # noqa: E402
    fried_parameter,
    greenwood_frequency,
    isoplanatic_angle,
)

LAM = 500e-9


def errors(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    """RMSE, MAE, bias and 95th-percentile |error|, all in dex (log10 units)."""
    e = pred - truth
    return (
        float(np.sqrt(np.mean(e**2))),
        float(np.mean(np.abs(e))),
        float(np.mean(e)),
        float(np.percentile(np.abs(e), 95)),
    )


def main() -> int:
    t_start = time.time()
    print("# CnCast — ML vs baseline benchmark (produced by validation/benchmark_ml.py)")
    print()
    print(f"- Python {platform.python_version()}, numpy {np.__version__}, "
          f"scikit-learn {sklearn.__version__}")
    print(f"- Platform: {platform.platform()}")
    print()

    t0 = time.time()
    model, art = train_default_model()
    t_train = time.time() - t0
    x_tr, y_tr = art["x_train"], art["y_train"]
    x_te, y_te = art["x_test"], art["y_test"]
    seeds = art["seeds"]

    print("## Setup")
    print()
    print(f"- scenarios: {len(art['fit_scenarios'])} fit + "
          f"{len(art['calibration_scenarios'])} calibration + "
          f"{len(art['test_scenarios'])} test (split by SCENARIO, seed "
          f"{seeds['split_seed']})")
    print(f"- rows: {x_tr.shape[0]} fit, {art['x_cal'].shape[0]} calibration, "
          f"{x_te.shape[0]} test ({art['n_altitudes']} altitudes per scenario)")
    print(f"- seeds: {seeds}")
    print(f"- fit + calibration wall time: **{t_train:.1f} s** on "
          f"{'2 cores (budget: 120 s)'}")
    print(f"- conformal delta applied to each interval bound: "
          f"{model.conformal_delta_dex:+.4f} dex")
    print(f"- quantile-crossing fraction on the fit set: "
          f"{model.fit_report()['quantile_crossing_fraction']:.4f}")
    print()

    # ---------------------------------------------------------------- table 1
    hv = Hv57Baseline()
    slc = SlcBaseline()
    clim = ClimatologyBaseline().fit(x_tr, y_tr)
    predictors = [
        ("CnCast learned model", model.predict_log10_cn2(x_te)),
        ("HV 5/7 (mandated baseline)", hv.predict_log10_cn2(x_te)),
        ("SLC day/night", slc.predict_log10_cn2(x_te)),
        ("Training climatology", clim.predict_log10_cn2(x_te)),
    ]
    print("## 1. Held-out error, all altitudes (units: dex = decades of Cn^2)")
    print()
    print("| predictor | RMSE | MAE | bias | p95 abs err |")
    print("|---|---:|---:|---:|---:|")
    for name, p in predictors:
        r, m, b, p95 = errors(p, y_te)
        print(f"| {name} | {r:.4f} | {m:.4f} | {b:+.4f} | {p95:.4f} |")
    print()
    ml_rmse = errors(predictors[0][1], y_te)[0]
    hv_rmse = errors(predictors[1][1], y_te)[0]
    print(f"Learned / HV 5/7 RMSE ratio: **{ml_rmse / hv_rmse:.3f}** "
          f"({(1 - ml_rmse / hv_rmse) * 100:.1f} % reduction).")
    print()

    # ---------------------------------------------------------------- table 2
    print("## 2. Held-out RMSE by altitude band (dex)")
    print()
    logh = x_te[:, 7]
    bands = [(5.0, 50.0), (50.0, 300.0), (300.0, 2000.0), (2000.0, 8000.0), (8000.0, 20000.0)]
    header = "| band [m] | n | " + " | ".join(n for n, _ in predictors) + " |"
    print(header)
    print("|---|---:|" + "---:|" * len(predictors))
    for lo, hi in bands:
        mask = (logh >= np.log10(lo)) & (logh < np.log10(hi) + (1e-12 if hi == 20000.0 else 0.0))
        cells = [f"{errors(p[mask], y_te[mask])[0]:.4f}" for _, p in predictors]
        print(f"| {lo:.0f}–{hi:.0f} | {int(mask.sum())} | " + " | ".join(cells) + " |")
    print()

    # ---------------------------------------------------------------- table 3
    print("## 3. Prediction-interval coverage on held-out data")
    print()
    lo_c, mid_c, hi_c = model._three(x_te)
    cov_c, wid_c = interval_coverage(y_te, lo_c, hi_c)

    raw = CnCastModel(random_state=seeds["model_random_state"]).fit(x_tr, y_tr)
    lo_r, _, hi_r = raw._three(x_te)
    cov_r, wid_r = interval_coverage(y_te, lo_r, hi_r)

    print("| interval | nominal | empirical coverage | mean width [dex] |")
    print("|---|---:|---:|---:|")
    print(f"| raw quantile GBR (alpha = 0.05 / 0.95) | 0.900 | {cov_r:.4f} | {wid_r:.4f} |")
    print(f"| conformally calibrated (CQR) | 0.900 | {cov_c:.4f} | {wid_c:.4f} |")
    print()
    n_te = y_te.size
    se = float(np.sqrt(cov_c * (1 - cov_c) / n_te))
    print(f"Binomial standard error on the calibrated coverage with n = {n_te} rows is "
          f"{se:.4f}; the rows are NOT independent (28 per scenario), so the effective "
          f"n is closer to the {len(art['test_scenarios'])} scenarios and the true "
          f"standard error is larger — treat +/-0.02 as the resolution of this estimate.")
    print()
    print("Coverage by altitude band (calibrated interval):")
    print()
    print("| band [m] | n | coverage | mean width [dex] |")
    print("|---|---:|---:|---:|")
    for lo, hi in bands:
        mask = (logh >= np.log10(lo)) & (logh < np.log10(hi) + (1e-12 if hi == 20000.0 else 0.0))
        c, w = interval_coverage(y_te[mask], lo_c[mask], hi_c[mask])
        print(f"| {lo:.0f}–{hi:.0f} | {int(mask.sum())} | {c:.4f} | {w:.4f} |")
    print()

    # ---------------------------------------------------------------- table 4
    print("## 4. Integrated seeing quantities from a predicted profile")
    print()
    sc = art["test_scenarios"][0]
    grid = default_altitude_grid(24)
    pred = model.predict_scenario(sc, grid)
    truth = profile_cn2(sc, grid)
    wind = bufton_wind(grid, sc.surface_wind_m_s)
    print(f"Test scenario 0: T = {sc.surface_temp_c:.2f} C, "
          f"wind = {sc.surface_wind_m_s:.2f} m/s, RH = {sc.relative_humidity_pct:.2f} %, "
          f"hour = {sc.hour_of_day:.2f}, day-of-year = {sc.day_of_year}")
    print()
    print("| quantity | from predicted median | from truth profile | from HV 5/7 |")
    print("|---|---:|---:|---:|")
    hv_grid = 10.0 ** hv.predict_log10_cn2(np.column_stack([np.zeros((grid.size, 7)),
                                                            np.log10(grid)]))
    rows = [
        ("r0 [cm]", lambda c: fried_parameter(grid, c, LAM, 0.0) * 100),
        ("theta0 [urad]", lambda c: isoplanatic_angle(grid, c, LAM, 0.0) * 1e6),
        ("f_G [Hz]", lambda c: greenwood_frequency(grid, c, wind, LAM, 0.0)),
    ]
    for label, fn in rows:
        print(f"| {label} | {fn(pred.cn2):.4f} | {fn(truth):.4f} | {fn(hv_grid):.4f} |")
    print()
    print("r0 from the interval bounds (upper Cn^2 -> smaller r0):")
    r0_lo = fried_parameter(grid, pred.cn2_lower, LAM, 0.0) * 100
    r0_hi = fried_parameter(grid, pred.cn2_upper, LAM, 0.0) * 100
    print(f"- r0(lower bound profile) = {r0_lo:.4f} cm")
    print(f"- r0(upper bound profile) = {r0_hi:.4f} cm")
    print()

    print("### Raw numbers for the hand check in VALIDATION.md §5")
    print()
    coarse = np.array([5.0, 100.0, 1000.0, 5000.0, 20000.0])
    pc = model.predict_scenario(sc, coarse)
    print("Predicted median Cn^2 on a 5-point coarse grid (same scenario):")
    print()
    print("| h [m] | Cn^2 [m^-2/3] |")
    print("|---:|---:|")
    for h, c in zip(coarse, pc.cn2, strict=True):
        print(f"| {h:.0f} | {c:.6e} |")
    print()
    r0_coarse = fried_parameter(coarse, pc.cn2, LAM, 0.0)
    print(f"- trapezoid mu_0 on this 5-point grid = "
          f"{float(np.trapezoid(pc.cn2, coarse)):.6e} m^(1/3)")
    print(f"- r0 on this 5-point grid = {r0_coarse * 100:.6f} cm "
          f"(the 24-point value above is the one to trust; the coarse grid exists only "
          f"so the arithmetic can be done by hand)")
    print()

    # ---------------------------------------------------------------- table 5
    print("## 5. Reproducibility")
    print()
    model2, art2 = train_default_model()
    same_data = bool(np.array_equal(art["x_test"], art2["x_test"]))
    p1 = model.predict_log10_cn2(x_te)
    p2 = model2.predict_log10_cn2(art2["x_test"])
    print(f"- identical test features on re-run: {same_data}")
    print(f"- max |prediction difference| across re-runs: {float(np.max(np.abs(p1 - p2))):.3e} dex")
    print(f"- conformal delta identical: "
          f"{model.conformal_delta_dex == model2.conformal_delta_dex}")
    print()
    print(f"Total script wall time: {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
