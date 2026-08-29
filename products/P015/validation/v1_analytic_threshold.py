"""V1 -- analytic optimal fixed threshold vs numerical and Monte Carlo optima.

Run from ``products/P015/``::

    PYTHONPATH=src python validation/v1_analytic_threshold.py

Three independent checks, all on **Scenario A** (stationary scintillation only,
noiseless telemetry, zero handover guard, RF always up), which is the regime in
which the closed forms of ``linkswitch.analytic`` are exact.

V1.1  Closed-form ``T*`` vs the numerical argmax of the bivariate-normal
      throughput expression ``expected_throughput_fixed_threshold``. These are
      two different derivations (a conditional-Gaussian indifference condition
      and a joint orthant probability), so agreement is a real check.
V1.2  Analytic throughput vs Monte Carlo throughput at a set of thresholds.
V1.3  Monte Carlo argmax under common random numbers vs the closed form.

All tolerances are stated before the numbers are printed and are never
relaxed; a violated tolerance is printed as FAILED.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, "src")

from linkswitch import (  # noqa: E402
    HybridLinkScenario,
    evaluate_selection,
    expected_throughput_fixed_threshold,
    optical_outage_probability,
    optimal_fixed_threshold_db,
    scenario_a_stationary,
    simulate_trace,
)

# Tolerances, fixed in advance.
TOL_NUMERIC_DB = 1.0e-3  # V1.1 closed form vs numerical argmax
TOL_MC_SIGMA = 4.0  # V1.2 |analytic - MC| in units of the MC standard error
TOL_MC_ARGMAX_DB = 0.20  # V1.3 closed form vs MC quadratic-fit argmax

N_MC_STEPS = 2_000_000  # samples for the throughput Monte Carlo
MC_SEED = 424242

results: list[tuple[str, bool]] = []


def report(label: str, passed: bool) -> None:
    """Record and print a pass/fail line."""
    results.append((label, passed))
    print(f"    -> {'PASS' if passed else 'FAILED'}: {label}")


def banner(text: str) -> None:
    """Print a section banner."""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    """Run all V1 checks; return 0 if every check passed."""
    t_start = time.time()
    sc = scenario_a_stationary()
    mu = sc.optical.margin_mean_db
    sd = sc.optical.margin_std_db
    rho = sc.rho

    banner("Scenario A -- stationary scintillation-only optical channel")
    print("  wavelength / path / wind      : 1.55 um / 3.0 km / 3.0 m/s")
    print(f"  scintillation index sigma_I^2 : {sc.optical.sigma_i2:.4f}")
    print(f"  sigma_ln (ln I)               : {sc.optical.sigma_ln:.6f}")
    print(f"  correlation time t_F          : {sc.optical.correlation_time_s * 1e3:.4f} ms")
    print(f"  sample interval dt            : {sc.dt_s * 1e3:.4f} ms")
    print(f"  lag-1 correlation rho         : {rho:.6f}")
    print(f"  margin mean mu                : {mu:.6f} dB")
    print(f"  margin std sigma              : {sd:.6f} dB")
    print(f"  optical rate R_o              : {sc.rate_optical_bps / 1e9:.3f} Gb/s")
    print(f"  RF rate R_r                   : {sc.rate_rf_bps / 1e6:.1f} Mb/s")
    print(f"  rate ratio q = R_r / R_o      : {sc.rate_ratio:.4f}")
    print(f"  analytic P(optical outage)    : {optical_outage_probability(mu, sd):.6f}")
    print(f"  handover guard                : {sc.switch_penalty_steps} samples")

    # ---------------------------------------------------------------- V1.1
    banner("V1.1  closed-form T* vs numerical argmax of the analytic throughput")
    print(f"  tolerance: |T*_closed - T*_numeric| <= {TOL_NUMERIC_DB:g} dB")
    print()
    print(f"  {'q':>8} {'T*_closed [dB]':>16} {'T*_numeric [dB]':>17} {'diff [dB]':>12}")
    worst = 0.0
    for q in (0.05, 0.10, 0.25, 0.40, 0.60):
        rate_rf = q * sc.rate_optical_bps
        t_closed = optimal_fixed_threshold_db(mu, sd, rho, sc.rate_optical_bps, rate_rf)

        def neg(t: float, _r: float = rate_rf) -> float:
            return -expected_throughput_fixed_threshold(
                t, mu, sd, rho, sc.rate_optical_bps, _r
            )

        opt = minimize_scalar(neg, bracket=(t_closed - 2.0, t_closed + 2.0), method="brent",
                              options={"xtol": 1e-10})
        diff = float(opt.x) - t_closed
        worst = max(worst, abs(diff))
        print(f"  {q:8.2f} {t_closed:16.6f} {float(opt.x):17.6f} {diff:12.3e}")
    print(f"\n  worst |diff| = {worst:.3e} dB")
    report(f"V1.1 closed form matches numerical argmax (worst {worst:.2e} dB)",
           worst <= TOL_NUMERIC_DB)

    # ---------------------------------------------------------------- V1.2
    banner("V1.2  analytic throughput vs Monte Carlo, at fixed thresholds")
    print(f"  Monte Carlo: {N_MC_STEPS:,} samples, seed {MC_SEED}, common random numbers")
    print(f"  tolerance: |analytic - MC| <= {TOL_MC_SIGMA:g} x MC standard error")
    print("  MC standard error uses the effective sample size n_eff = n (1-rho)/(1+rho)")
    print("  (Bartlett's variance inflation for an AR(1) series; Bartlett, JRSS Suppl. 8, 1946)")
    big = HybridLinkScenario(
        optical=sc.optical, rf=sc.rf, dt_s=sc.dt_s, n_steps=N_MC_STEPS,
        rate_optical_bps=sc.rate_optical_bps, rate_rf_bps=sc.rate_rf_bps,
        switch_penalty_steps=0,
    )
    trace = simulate_trace(big, MC_SEED)
    tele_prev = np.empty(N_MC_STEPS)
    tele_prev[0] = trace.optical_telemetry_db[0]
    tele_prev[1:] = trace.optical_telemetry_db[:-1]
    n_eff = N_MC_STEPS * (1.0 - rho) / (1.0 + rho)
    print(f"  n_eff = {n_eff:,.0f}")
    print()
    print(f"  {'T [dB]':>9} {'analytic [Mb/s]':>17} {'MC [Mb/s]':>13} "
          f"{'diff [Mb/s]':>13} {'diff/SE':>9}")
    t_star = optimal_fixed_threshold_db(mu, sd, rho, sc.rate_optical_bps, sc.rate_rf_bps)
    worst_sigma = 0.0
    for t in (-4.0, -3.0, t_star, -1.0, 0.0, 1.0, 2.0):
        sel = tele_prev >= t
        m = evaluate_selection(
            sel, trace.optical_up, trace.rf_up,
            rate_optical_bps=big.rate_optical_bps, rate_rf_bps=big.rate_rf_bps,
            dt_s=big.dt_s, switch_penalty_steps=0,
        )
        rate = np.where(sel, np.where(trace.optical_up, big.rate_optical_bps, 0.0),
                        np.where(trace.rf_up, big.rate_rf_bps, 0.0))
        se = float(rate.std(ddof=1)) / np.sqrt(n_eff)
        ana = expected_throughput_fixed_threshold(
            t, mu, sd, rho, big.rate_optical_bps, big.rate_rf_bps
        )
        d = ana - m.throughput_bps
        worst_sigma = max(worst_sigma, abs(d) / se)
        print(f"  {t:9.4f} {ana / 1e6:17.4f} {m.throughput_bps / 1e6:13.4f} "
              f"{d / 1e6:13.4f} {d / se:9.2f}")
    print(f"\n  worst |diff|/SE = {worst_sigma:.2f}")
    report(f"V1.2 analytic throughput matches MC (worst {worst_sigma:.2f} sigma)",
           worst_sigma <= TOL_MC_SIGMA)

    # ---------------------------------------------------------------- V1.3
    banner("V1.3  Monte Carlo argmax (common random numbers) vs closed form")
    print(f"  tolerance: |T*_closed - T*_MC| <= {TOL_MC_ARGMAX_DB:g} dB")
    print("  T*_MC from a quadratic least-squares fit to the MC throughput curve over")
    print("  T* +/- 0.6 dB (step 0.05 dB); common random numbers make the *differences*")
    print("  between thresholds far less noisy than the absolute throughputs.")
    print()
    print(f"  {'q':>8} {'T*_closed [dB]':>16} {'T*_MC [dB]':>13} {'diff [dB]':>11} "
          f"{'gain vs always-opt [Mb/s]':>27}")
    worst_mc = 0.0
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for q in (0.10, 0.25, 0.40):
        rate_rf = q * big.rate_optical_bps
        t_closed = optimal_fixed_threshold_db(mu, sd, rho, big.rate_optical_bps, rate_rf)
        grid = np.arange(t_closed - 0.6, t_closed + 0.6001, 0.05)
        tp = np.empty(grid.size)
        for i, t in enumerate(grid):
            sel = tele_prev >= t
            rate = np.where(sel, np.where(trace.optical_up, big.rate_optical_bps, 0.0),
                            np.where(trace.rf_up, rate_rf, 0.0))
            tp[i] = rate.mean()
        curves[q] = (grid, tp)
        coef = np.polyfit(grid - t_closed, tp, 2)
        t_mc = t_closed - coef[1] / (2.0 * coef[0])
        always_opt = float(
            np.where(trace.optical_up, big.rate_optical_bps, 0.0).mean()
        )
        gain = tp.max() - always_opt
        d = t_mc - t_closed
        worst_mc = max(worst_mc, abs(d))
        print(f"  {q:8.2f} {t_closed:16.6f} {t_mc:13.6f} {d:11.4f} {gain / 1e6:27.3f}")
    print(f"\n  worst |diff| = {worst_mc:.4f} dB")
    report(f"V1.3 MC argmax matches closed form (worst {worst_mc:.4f} dB)",
           worst_mc <= TOL_MC_ARGMAX_DB)

    # ------------------------------------------------------- context
    banner("Context: curvature of the throughput curve around T*")
    grid, tp = curves[0.25]
    t_closed = optimal_fixed_threshold_db(mu, sd, rho, big.rate_optical_bps, big.rate_rf_bps)
    print("  q = 0.25; throughput penalty for mis-setting the threshold (analytic):")
    print(f"  {'offset [dB]':>12} {'E[R] [Mb/s]':>14} {'loss vs T* [Mb/s]':>19}")
    e_star = expected_throughput_fixed_threshold(
        t_closed, mu, sd, rho, big.rate_optical_bps, big.rate_rf_bps
    )
    for off in (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0):
        e = expected_throughput_fixed_threshold(
            t_closed + off, mu, sd, rho, big.rate_optical_bps, big.rate_rf_bps
        )
        print(f"  {off:12.2f} {e / 1e6:14.4f} {(e_star - e) / 1e6:19.4f}")

    banner("SUMMARY")
    for label, ok in results:
        print(f"  {'PASS  ' if ok else 'FAILED'}  {label}")
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n  {len(results) - n_fail}/{len(results)} checks passed")
    print(f"  wall time: {time.time() - t_start:.1f} s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
