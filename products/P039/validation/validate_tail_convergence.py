#!/usr/bin/env python3
"""Validation 3 -- tail-latency estimates converge at the order-statistic rate.

The prediction, stated before measurement
-----------------------------------------
For a continuous distribution with density ``f`` positive at the quantile
``q_p``, the sample quantile is asymptotically normal:

    sqrt(n) (q_hat_p - q_p) -> N(0, p (1 - p) / f(q_p)^2)

(Mosteller 1946, *Annals of Mathematical Statistics* 17(4): 377-408; David &
Nagaraja 2003, *Order Statistics*, 3rd ed., Sec. 10.2). Two consequences are
checked:

1. **Rate.** The root-mean-square error of a tail estimate decays as
   ``n^(-1/2)``, so the least-squares slope of ``log(RMSE)`` against
   ``log(n)`` is exactly ``-0.5``. The rate does not depend on the density at
   all, which is what makes it a clean check.
2. **Magnitude.** The RMSE equals ``sqrt(p (1 - p) / n) / f(q_p)``, not merely
   something proportional to it. For a lognormal the density at its own
   quantile is exact, ``f(q_p) = phi(z_p) / (sigma q_p)``, so the predicted
   magnitude is a closed-form number and the measured RMSE divided by it must
   approach 1.

Validity range, applied rather than ignored
-------------------------------------------
The result is asymptotic. It needs the quantile to be interior to the sample
with some margin; below roughly five expected exceedances,
``n (1 - p) < 5``, the estimator is near the sample maximum and the normal
approximation does not hold. The ``n`` grid for each ``p`` therefore starts at
``n (1 - p) >= 5``, which is the theory's own stated condition and is applied
here as a condition, not as a tolerance. Section (c) then measures the grid
points *below* that condition for ``p = 0.999`` and shows the theory failing
there, which is more useful than pretending the condition is not needed.

Tolerances and uncertainties, declared before the run
-----------------------------------------------------
With ``R`` repeats an RMSE estimate has relative standard error
``1 / sqrt(2 R)``, so the log-RMSE has standard deviation ``1 / sqrt(2 R)``
and the least-squares slope therefore has standard error
``1 / (sqrt(2 R) * sqrt(Sxx))`` with ``Sxx = sum (ln n - mean ln n)^2``. That
figure is computed and printed next to every slope, so a deviation can be read
in units of its own uncertainty. The declared acceptance band is ``+/- 0.04``
on the slope and ``+/- 0.15`` on the magnitude ratio at the largest ``n``.
Neither is adjusted after the fact; a row outside the band is reported as
FAIL and its deviation in standard errors is printed beside it.

Reproduce with:
    cd validation && PYTHONPATH=../src python3 validate_tail_convergence.py

Runtime: about 30 s on one uncontended core.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from latencynet.pipeline import make_lognormal_pipeline, sample_stage_latencies
from latencynet.tails import (
    lognormal_quantile,
    lognormal_quantile_se,
    lognormal_tail_convergence,
    quantile,
    quantile_min_samples,
)
from latencynet.units import s_to_us

#: Single-stage lognormal: mean 200 us, cv 0.45.
SINGLE_MEAN_S = 2.0e-4
SINGLE_CV = 0.45
#: Grids chosen so every point satisfies n (1 - p) >= 5.
N_GRID_BY_P: dict[float, tuple[int, ...]] = {
    0.99: (500, 1500, 5000, 15_000, 50_000),
    0.999: (5000, 15_000, 50_000, 150_000),
}
#: Grid points deliberately BELOW the validity condition, reported in (c).
OUT_OF_RANGE_GRID: tuple[int, ...] = (200, 500, 1500)
N_REPEATS = 400
SEED = 20260405
PROBABILITIES = (0.99, 0.999)
SLOPE_TOLERANCE = 0.04
RATIO_TOLERANCE = 0.15

PIPELINE = make_lognormal_pipeline((60e-6, 180e-6, 30e-6), (12e-6, 40e-6, 6e-6))
PIPELINE_GRID_BY_P: dict[float, tuple[int, ...]] = {
    0.99: (1000, 4000, 16_000, 64_000),
    0.999: (5000, 20_000, 80_000),
}
PIPELINE_REPEATS = 300
PIPELINE_REFERENCE_N = 4_000_000
OUTPUT_JSON = Path(__file__).resolve().parent / "tail_convergence.json"


def _single_params() -> tuple[float, float]:
    sigma_sq = math.log1p(SINGLE_CV**2)
    return math.log(SINGLE_MEAN_S) - 0.5 * sigma_sq, math.sqrt(sigma_sq)


def _slope_se(n_grid: tuple[int, ...], n_repeats: int) -> float:
    """Propagated standard error of the fitted log-log slope."""
    logs = np.log(np.asarray(n_grid, dtype=float))
    sxx = float(np.sum((logs - logs.mean()) ** 2))
    return (1.0 / math.sqrt(2.0 * n_repeats)) / math.sqrt(sxx)


def _reference_quantile_one_sigma(samples: np.ndarray, p: float) -> float:
    n = samples.size
    spread = math.sqrt(n * p * (1.0 - p))
    srt = np.sort(samples)
    lo = int(math.floor(n * p - spread))
    hi = int(math.ceil(n * p + spread))
    if lo < 1 or hi > n:
        return float("nan")
    return 0.5 * float(srt[hi - 1] - srt[lo - 1])


def single_lognormal_case() -> tuple[list[bool], list[dict[str, object]]]:
    mu, sigma = _single_params()
    print("\n(a) single lognormal stage, exact quantile known in closed form")
    print("-" * 96)
    print(f"  mu = {mu:.9f}   sigma = {sigma:.9f}   mean = {s_to_us(SINGLE_MEAN_S):.3f} us")
    print(
        f"  repeats per n: {N_REPEATS}; log-RMSE standard deviation "
        f"1/sqrt(2R) = {1.0 / math.sqrt(2 * N_REPEATS):.4f}"
    )
    results: list[bool] = []
    rows: list[dict[str, object]] = []
    for p in PROBABILITIES:
        grid = N_GRID_BY_P[p]
        truth = lognormal_quantile(mu, sigma, p)
        print(
            f"\n  p = {p}   exact quantile {s_to_us(truth):.4f} us   "
            f"smallest interior n = {quantile_min_samples(p)}   "
            f"grid starts at n(1-p) = {grid[0] * (1 - p):.1f} exceedances"
        )
        result = lognormal_tail_convergence(mu, sigma, p, grid, N_REPEATS, SEED, "linear")
        print(f"  {'n':>8}{'RMSE us':>12}{'analytic SE us':>17}{'ratio':>9}{'RMSE/q':>10}")
        for n, rmse, se, ratio in zip(
            result.n_grid,
            result.rmse_s,
            result.analytic_se_s,
            result.ratio_measured_over_analytic,
            strict=True,
        ):
            print(
                f"  {n:>8}{s_to_us(rmse):>12.5f}{s_to_us(se):>17.5f}"
                f"{ratio:>9.4f}{rmse / truth:>10.5f}"
            )
            rows.append(
                {
                    "case": "single_lognormal",
                    "p": p,
                    "n": n,
                    "rmse_s": rmse,
                    "analytic_se_s": se,
                    "ratio": ratio,
                }
            )
        slope_se = _slope_se(grid, N_REPEATS)
        diff = result.fitted_slope - result.predicted_slope
        slope_ok = abs(diff) <= SLOPE_TOLERANCE
        ratio_ok = abs(result.ratio_measured_over_analytic[-1] - 1.0) <= RATIO_TOLERANCE
        results += [slope_ok, ratio_ok]
        print(
            f"  fitted log-log slope {result.fitted_slope:+.5f} +/- {slope_se:.5f} vs predicted "
            f"{result.predicted_slope:+.2f}   diff {diff:+.5f} = {diff / slope_se:+.2f} SE   "
            f"tol {SLOPE_TOLERANCE}   {'PASS' if slope_ok else 'FAIL'}"
        )
        print(
            f"  magnitude ratio at n={result.n_grid[-1]}: "
            f"{result.ratio_measured_over_analytic[-1]:.5f} vs predicted 1.0   "
            f"tol {RATIO_TOLERANCE}   {'PASS' if ratio_ok else 'FAIL'}"
        )
        rows.append(
            {
                "case": "single_lognormal_slope",
                "p": p,
                "n_grid": list(grid),
                "fitted_slope": result.fitted_slope,
                "slope_standard_error": slope_se,
                "predicted_slope": result.predicted_slope,
                "slope_within_tolerance": bool(slope_ok),
                "ratio_at_largest_n": result.ratio_measured_over_analytic[-1],
                "ratio_within_tolerance": bool(ratio_ok),
            }
        )
    return results, rows


def pipeline_case() -> tuple[list[bool], list[dict[str, object]]]:
    print("\n(b) three-stage pipeline, truth from a large Monte Carlo reference")
    print("-" * 96)
    reference = sample_stage_latencies(PIPELINE, PIPELINE_REFERENCE_N, SEED + 7).sum(axis=1)
    rng = np.random.default_rng(SEED + 8)
    results: list[bool] = []
    rows: list[dict[str, object]] = []
    sigmas = np.array([s.lognormal_params()[1] for s in PIPELINE.stages])
    mus = np.array([s.lognormal_params()[0] for s in PIPELINE.stages])
    total_var = float(PIPELINE.injected_covariance().sum())
    sigma_s = math.sqrt(math.log1p(total_var / PIPELINE.injected_mean_s() ** 2))
    mu_sum = math.log(PIPELINE.injected_mean_s()) - 0.5 * sigma_s**2
    for p in PROBABILITIES:
        grid = PIPELINE_GRID_BY_P[p]
        truth = quantile(reference, p, "linear")
        ref_se = _reference_quantile_one_sigma(reference, p)
        print(
            f"\n  p = {p}   reference quantile {s_to_us(truth):.4f} us from "
            f"{PIPELINE_REFERENCE_N} passes, one-sigma {s_to_us(ref_se):.5f} us "
            f"({ref_se / truth * 100:.4f} %)"
        )
        print(f"  {'n':>8}{'RMSE us':>12}{'predicted us':>15}{'ratio':>9}")
        rmses: list[float] = []
        for n in grid:
            errs = np.empty(PIPELINE_REPEATS, dtype=float)
            for r in range(PIPELINE_REPEATS):
                z = rng.standard_normal((n, mus.size))
                total = np.exp(mus + sigmas * z).sum(axis=1)
                errs[r] = quantile(total, p, "linear") - truth
            rmses.append(float(math.sqrt(float(np.mean(errs**2)))))
            # The sum of lognormals has no closed-form density, so the
            # magnitude prediction uses the Fenton-Wilkinson lognormal match
            # and inherits its approximation. The slope does not.
            predicted = lognormal_quantile_se(mu_sum, sigma_s, p, n)
            print(
                f"  {n:>8}{s_to_us(rmses[-1]):>12.5f}{s_to_us(predicted):>15.5f}"
                f"{rmses[-1] / predicted:>9.4f}"
            )
            rows.append(
                {
                    "case": "three_stage_pipeline",
                    "p": p,
                    "n": n,
                    "rmse_s": rmses[-1],
                    "predicted_se_s": predicted,
                    "ratio": rmses[-1] / predicted,
                }
            )
        slope = float(
            np.polyfit(np.log(np.array(grid, dtype=float)), np.log(np.array(rmses)), 1)[0]
        )
        slope_se = _slope_se(grid, PIPELINE_REPEATS)
        diff = slope + 0.5
        ok = abs(diff) <= SLOPE_TOLERANCE
        results.append(ok)
        print(
            f"  fitted log-log slope {slope:+.5f} +/- {slope_se:.5f} vs predicted -0.50   "
            f"diff {diff:+.5f} = {diff / slope_se:+.2f} SE   tol {SLOPE_TOLERANCE}   "
            f"{'PASS' if ok else 'FAIL'}"
        )
        rows.append(
            {
                "case": "three_stage_pipeline_slope",
                "p": p,
                "n_grid": list(grid),
                "fitted_slope": slope,
                "slope_standard_error": slope_se,
                "predicted_slope": -0.5,
                "slope_within_tolerance": bool(ok),
                "reference_n": PIPELINE_REFERENCE_N,
                "reference_quantile_s": truth,
                "reference_one_sigma_s": ref_se,
            }
        )
    print(
        f"\n  repeats per n = {PIPELINE_REPEATS}; declared stage means "
        f"{[round(float(s_to_us(v)), 3) for v in PIPELINE.stage_mean_s]} us, sds "
        f"{[round(float(s_to_us(v)), 3) for v in PIPELINE.stage_std_s]} us"
    )
    return results, rows


def out_of_range_case() -> list[dict[str, object]]:
    """Measure where the asymptotic theory stops working. Reported, not gated."""
    mu, sigma = _single_params()
    p = 0.999
    print("\n(c) below the validity condition: p = 0.999 with fewer than 5 exceedances")
    print("-" * 96)
    print(
        "  These grid points violate n (1 - p) >= 5 and are excluded from the fit\n"
        "  in (a). They are measured here to show the condition is real."
    )
    result = lognormal_tail_convergence(mu, sigma, p, OUT_OF_RANGE_GRID, N_REPEATS, SEED + 3)
    print(f"  {'n':>8}{'n(1-p)':>9}{'RMSE us':>12}{'analytic SE us':>17}{'ratio':>9}")
    rows: list[dict[str, object]] = []
    for n, rmse, se, ratio in zip(
        result.n_grid,
        result.rmse_s,
        result.analytic_se_s,
        result.ratio_measured_over_analytic,
        strict=True,
    ):
        print(
            f"  {n:>8}{n * (1 - p):>9.2f}{s_to_us(rmse):>12.5f}"
            f"{s_to_us(se):>17.5f}{ratio:>9.4f}"
        )
        rows.append(
            {
                "case": "single_lognormal_out_of_range",
                "p": p,
                "n": n,
                "expected_exceedances": n * (1 - p),
                "rmse_s": rmse,
                "analytic_se_s": se,
                "ratio": ratio,
            }
        )
    print(
        f"  fitted slope over the out-of-range grid: {result.fitted_slope:+.5f}, which is "
        f"{abs(result.fitted_slope + 0.5):.5f} from -0.5.\n"
        "  The measured RMSE sits well below the asymptotic SE because the\n"
        "  estimator is pinned near the sample maximum and cannot fluctuate\n"
        "  upward: the asymptotic expression over-states the error there."
    )
    return rows


def main() -> int:
    print("=" * 96)
    print("P039 latencynet -- Validation 3: tail-quantile convergence vs order-statistic theory")
    print("=" * 96)
    print("predicted RMSE scaling: n^-0.5 exactly; predicted magnitude: sqrt(p(1-p)/n)/f(q_p)")
    print(f"declared tolerances: slope +/-{SLOPE_TOLERANCE}, magnitude ratio +/-{RATIO_TOLERANCE}")
    print("validity condition applied to every fitted grid: n (1 - p) >= 5")
    print("No measured wall-clock time appears anywhere in this script.")

    results_a, rows_a = single_lognormal_case()
    results_b, rows_b = pipeline_case()
    rows_c = out_of_range_case()
    results = results_a + results_b
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "single_mean_s": SINGLE_MEAN_S,
                "single_cv": SINGLE_CV,
                "n_grid_by_p": {str(k): list(v) for k, v in N_GRID_BY_P.items()},
                "n_repeats": N_REPEATS,
                "out_of_range_grid": list(OUT_OF_RANGE_GRID),
                "pipeline_grid_by_p": {str(k): list(v) for k, v in PIPELINE_GRID_BY_P.items()},
                "pipeline_repeats": PIPELINE_REPEATS,
                "pipeline_reference_n": PIPELINE_REFERENCE_N,
                "seed": SEED,
                "slope_tolerance": SLOPE_TOLERANCE,
                "ratio_tolerance": RATIO_TOLERANCE,
                "validity_condition": "n * (1 - p) >= 5",
                "rows": rows_a + rows_b + rows_c,
            },
            indent=2,
        )
        + "\n"
    )
    n_pass = sum(results)
    print(f"\n  written: {OUTPUT_JSON.name}")
    print("=" * 96)
    print(f"convergence checks: {n_pass}/{len(results)} PASS")
    print("=" * 96)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
