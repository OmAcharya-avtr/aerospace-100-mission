"""Validation 2 — NEES/NIS inside chi-squared bounds when correct, provably outside when not.

Run from the product root::

    PYTHONPATH=src python3 validation/v2_nees_nis_consistency.py

METHOD.  ``M`` independent Monte Carlo runs of a 1-D CWNA constant-velocity
truth (state ``[position m, velocity m/s]``) with position-only measurements.
For each run the truth, the process noise and the measurement noise are drawn
from the *same* statistics the filter assumes (Part A) or from deliberately
different ones (Part B).

At every step the ensemble average

    ANEES_k = (1/M) Σ_i  x̃ᵢₖᵀ Pᵢₖ⁻¹ x̃ᵢₖ          ANIS_k likewise with ν, S

is compared with the two-sided 95 % acceptance region ``[χ²_{M d}(0.025)/M,
χ²_{M d}(0.975)/M]`` (Bar-Shalom, Li & Kirubarajan 2001, *Estimation with
Applications to Tracking and Navigation*, Eq. (5.4.2-3)), with ``d = 2`` for
NEES and ``d = 1`` for NIS.  Runs are independent by construction (seed
``base + i``), so the independence assumption behind the bound holds.

The first ``BURN_IN`` steps are excluded: the transient from an intentionally
loose ``P₀`` is a modelling choice, not an inconsistency.

Part C additionally applies the innovation whiteness test
(Bar-Shalom et al. Eq. (5.4.3-2), band ±1.96/√N at 95 %) to the pooled
innovation sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navbench import (  # noqa: E402
    KalmanFilter,
    chi2_bounds,
    constant_velocity_cwna,
    ensemble_consistency,
    innovation_whiteness,
    nees,
    nis,
    simulate_linear_system,
)

DT = 1.0
Q_PSD_TRUE = 0.05  # m^2/s^3
SIGMA_Z = 3.0  # m
N_STEPS = 200
N_RUNS = 60
BURN_IN = 30
BASE_SEED = 90210


def run_ensemble(q_factor: float, r_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-run NEES, NIS and pooled innovations for a given mis-specification."""
    f, q_true = constant_velocity_cwna(DT, Q_PSD_TRUE)
    _, q_filter = constant_velocity_cwna(DT, Q_PSD_TRUE * q_factor)
    h = np.array([[1.0, 0.0]])
    r_true = np.array([[SIGMA_Z**2]])
    r_filter = np.array([[SIGMA_Z**2 * r_factor]])
    p0 = np.diag([100.0, 10.0])
    nees_runs = np.zeros((N_RUNS, N_STEPS))
    nis_runs = np.zeros((N_RUNS, N_STEPS))
    innov = np.zeros((N_RUNS, N_STEPS))
    for i in range(N_RUNS):
        rng = np.random.default_rng(BASE_SEED + i)
        truth, meas = simulate_linear_system(
            f, h, q_true, r_true, np.array([0.0, 1.0]), N_STEPS, rng
        )
        kf = KalmanFilter(f, h, q_filter, r_filter, np.array([0.0, 0.0]), p0)
        res = kf.run(meas)
        nees_runs[i] = nees(truth - res.x_post, res.p_post)
        nis_runs[i] = nis(res.innovation, res.innovation_cov)
        innov[i] = res.innovation[:, 0]
    return nees_runs, nis_runs, innov


def report(label: str, q_factor: float, r_factor: float) -> dict[str, float]:
    nees_runs, nis_runs, _ = run_ensemble(q_factor, r_factor)
    a_nees, lo_e, hi_e = ensemble_consistency(nees_runs[:, BURN_IN:], 2)
    a_nis, lo_i, hi_i = ensemble_consistency(nis_runs[:, BURN_IN:], 1)
    frac_e = float(np.mean((a_nees >= lo_e) & (a_nees <= hi_e)))
    frac_i = float(np.mean((a_nis >= lo_i) & (a_nis <= hi_i)))
    print(f"\n  {label}")
    print(f"    Q factor {q_factor:g}, R factor {r_factor:g}")
    print(
        f"    ANEES: mean {np.mean(a_nees):.4f}  (dof 2, expectation 2.0), "
        f"bounds [{lo_e:.4f}, {hi_e:.4f}], {100.0 * frac_e:5.1f} % of steps inside"
    )
    print(
        f"    ANIS : mean {np.mean(a_nis):.4f}  (dof 1, expectation 1.0), "
        f"bounds [{lo_i:.4f}, {hi_i:.4f}], {100.0 * frac_i:5.1f} % of steps inside"
    )
    return {
        "anees": float(np.mean(a_nees)),
        "anis": float(np.mean(a_nis)),
        "frac_nees": frac_e,
        "frac_nis": frac_i,
        "lo_e": lo_e,
        "hi_e": hi_e,
        "lo_i": lo_i,
        "hi_i": hi_i,
    }


def main() -> int:
    print("=" * 78)
    print("v2 - NEES / NIS chi-squared consistency")
    print("=" * 78)
    print(
        f"  {N_RUNS} independent runs x {N_STEPS} steps, dt = {DT} s, "
        f"q_psd_true = {Q_PSD_TRUE} m^2/s^3, sigma_z = {SIGMA_Z} m, burn-in {BURN_IN}"
    )
    print(f"  single-sample 95 % NEES band (dof 2): {chi2_bounds(2, 1)}")
    print(f"  ensemble 95 % NEES band  (M = {N_RUNS}): {chi2_bounds(2, N_RUNS)}")
    print(f"  ensemble 95 % NIS  band  (M = {N_RUNS}): {chi2_bounds(1, N_RUNS)}")

    print("\n" + "-" * 78)
    print("PART A - correctly specified filter (must be INSIDE the bounds)")
    print("-" * 78)
    a = report("correct: Q_filter = Q_true, R_filter = R_true", 1.0, 1.0)
    ok_a = a["lo_e"] <= a["anees"] <= a["hi_e"] and a["lo_i"] <= a["anis"] <= a["hi_i"]
    print(f"    verdict: {'PASS' if ok_a else 'FAIL'} (both averages inside their bounds)")

    print("\n" + "-" * 78)
    print("PART B - deliberately mis-specified filters (must LEAVE the bounds)")
    print("-" * 78)
    cases = [
        ("Q too small by 25x  -> expect OPTIMISTIC (ANEES above)", 1.0 / 25.0, 1.0, "high"),
        ("Q too large by 25x  -> expect PESSIMISTIC (ANEES below)", 25.0, 1.0, "low"),
        ("R too small by 9x   -> expect OPTIMISTIC (ANIS above)", 1.0, 1.0 / 9.0, "high"),
        ("R too large by 9x   -> expect PESSIMISTIC (ANIS below)", 1.0, 9.0, "low"),
    ]
    ok_b = True
    for label, qf, rf, direction in cases:
        res = report(label, qf, rf)
        if direction == "high":
            got = res["anees"] > res["hi_e"] or res["anis"] > res["hi_i"]
        else:
            got = res["anees"] < res["lo_e"] or res["anis"] < res["lo_i"]
        ok_b &= got
        print(
            f"    verdict: {'PASS' if got else 'FAIL'} "
            f"(left the bounds in the {direction} direction)"
        )

    print("\n" + "-" * 78)
    print("PART C - innovation whiteness (Bar-Shalom et al. Eq. 5.4.3-2)")
    print("-" * 78)
    ok_c = True
    for label, qf, rf, expect_white in (
        ("correct filter", 1.0, 1.0, True),
        ("Q too small by 25x", 1.0 / 25.0, 1.0, False),
    ):
        _, _, innov = run_ensemble(qf, rf)
        w = innovation_whiteness(innov[0, BURN_IN:], max_lag=10)
        print(f"\n  {label}: {w.summary()}")
        print(
            "    rho(1..5) = "
            + ", ".join(f"{v:+.4f}" for v in w.autocorrelation[1:6])
        )
        got = w.passed == expect_white
        ok_c &= got
        print(
            f"    expected {'white' if expect_white else 'correlated'} -> "
            f"{'PASS' if got else 'FAIL'}"
        )

    overall = ok_a and ok_b and ok_c
    print()
    print("=" * 78)
    print(f"OVERALL v2: {'PASS' if overall else 'FAIL'}")
    print("=" * 78)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
