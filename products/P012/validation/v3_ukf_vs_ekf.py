"""Validation 3 — UKF matches EKF when nearly linear, degrades more gracefully when not.

Run from the product root::

    PYTHONPATH=src python3 validation/v3_ukf_vs_ekf.py

SCENARIO.  2-D constant-velocity target, ``x = [x, vx, y, vy]`` in m and m/s,
observed in polar coordinates by a sensor at the origin:
``h(x) = [sqrt(x²+y²), atan2(y, x)]``.  The nonlinearity of ``h`` is governed
by the ratio of the position uncertainty to the range: ``∂θ/∂r`` scales as
``1/r``, so a target far away with a small covariance is nearly linear and a
target passing close with a large covariance is strongly nonlinear.

PART A — near-linear regime.  Range ≈ 11 km, ``σ_range = 20 m``,
``σ_bearing = 5 mrad``, ``q̃ = 0.05 m²/s³``.  The EKF and the UKF (α = 1,
β = 2, κ = 0) must agree.  "Agree" is defined *before* the run as: total
position RMSE within 1 % of each other, and per-step state difference below
1 % of the filter's own reported position standard deviation.  Both are
reported as measured.

PART B — strongly nonlinear regime.  Target passes within ~120 m of the
sensor, ``σ_range = 60 m``, ``σ_bearing = 0.35 rad`` (20°), ``q̃ = 5 m²/s³``.
40 independent Monte Carlo runs.  Reported per filter: mean and median
position RMSE, mean NEES, and the number of runs whose terminal NEES exceeds
the ``χ²₄`` 99.99 % quantile (the divergence convention of
:mod:`navbench.bench`).

PART C — control.  With a *linear* measurement matrix ``H = [[1,0,0,0],
[0,0,1,0]]`` and identical noise, the EKF and the UKF must both reduce to the
linear Kalman filter to round-off.  This establishes that any difference seen
in Part B is attributable to the nonlinearity and not to an implementation
difference between the two filters.

Julier & Uhlmann (2004), *Proceedings of the IEEE* 92(3), 401-422, §V is the
source of the claim under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy import stats  # noqa: E402

from navbench import (  # noqa: E402
    ExtendedKalmanFilter,
    KalmanFilter,
    UnscentedKalmanFilter,
    constant_velocity_2d,
    nees,
    radar_jacobian,
    radar_measurement,
    simulate_linear_system,
    simulate_radar_scenario,
)

DT = 1.0


def _make_filters(f, q, r, x_init, p0, alpha=1.0):
    ekf = ExtendedKalmanFilter(
        lambda x: f @ x, radar_measurement, q, r, x_init, p0,
        f_jac=lambda x: f, h_jac=radar_jacobian,
    )
    ukf = UnscentedKalmanFilter(
        lambda x: f @ x, radar_measurement, q, r, x_init, p0,
        alpha=alpha, beta=2.0, kappa=0.0,
    )
    return ekf, ukf


def part_a() -> bool:
    print("=" * 78)
    print("PART A - near-linear regime: EKF and UKF must agree")
    print("=" * 78)
    x0 = np.array([8000.0, -5.0, 8000.0, 3.0])
    q_psd, sr, sb, n_steps = 0.05, 20.0, 0.005, 200
    f, q = constant_velocity_2d(DT, q_psd)
    r = np.diag([sr**2, sb**2])
    p0 = np.diag([300.0, 50.0, 300.0, 50.0])
    rng = np.random.default_rng(31415)
    truth, meas = simulate_radar_scenario(
        dt=DT, n_steps=n_steps, q_psd=q_psd, sigma_range=sr, sigma_bearing=sb, x0=x0, rng=rng
    )
    x_init = np.array([truth[0, 0] + 10.0, 0.0, truth[0, 2] + 10.0, 0.0])
    ekf, ukf = _make_filters(f, q, r, x_init, p0)
    res_e, res_u = ekf.run(meas), ukf.run(meas)

    rmse_e = float(np.sqrt(np.mean((truth[:, [0, 2]] - res_e.x_post[:, [0, 2]]) ** 2)))
    rmse_u = float(np.sqrt(np.mean((truth[:, [0, 2]] - res_u.x_post[:, [0, 2]]) ** 2)))
    rel_rmse = abs(rmse_e - rmse_u) / rmse_e
    sig_pos = np.sqrt(res_e.p_post[:, 0, 0])
    dx = np.abs(res_e.x_post - res_u.x_post)
    rel_state = float(np.max(dx[:, [0, 2]] / sig_pos[:, None]))
    rel_cov = float(
        np.max(np.abs(res_e.p_post - res_u.p_post)) / np.max(np.abs(res_e.p_post))
    )
    mean_range = float(np.mean(np.hypot(truth[:, 0], truth[:, 2])))
    print(f"  mean range {mean_range:.1f} m, sigma_range {sr} m, sigma_bearing {sb} rad")
    print(f"  cross-range measurement resolution r*sigma_b = {mean_range * sb:.1f} m")
    print(f"  EKF position RMSE                : {rmse_e:.9f} m")
    print(f"  UKF position RMSE                : {rmse_u:.9f} m")
    print(f"  relative RMSE difference         : {rel_rmse:.3e}   (tolerance 1e-2)")
    print(f"  max |x_EKF - x_UKF| / sigma_pos  : {rel_state:.3e}   (tolerance 1e-2)")
    print(f"  max |P_EKF - P_UKF| / max|P_EKF| : {rel_cov:.3e}")
    ok = rel_rmse < 1e-2 and rel_state < 1e-2
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def part_b() -> bool:
    print()
    print("=" * 78)
    print("PART B - strongly nonlinear regime: UKF must degrade more gracefully")
    print("=" * 78)
    x0 = np.array([600.0, -20.0, 120.0, -4.0])
    q_psd, sr, sb, n_steps, n_runs = 5.0, 60.0, 0.35, 60, 40
    f, q = constant_velocity_2d(DT, q_psd)
    r = np.diag([sr**2, sb**2])
    p0 = np.diag([300.0, 50.0, 300.0, 50.0])
    thresh = float(stats.chi2.ppf(0.9999, 4))
    stats_out: dict[str, dict[str, float]] = {}
    min_range = None
    for name in ("EKF", "UKF"):
        rmses, neeses, diverged = [], [], 0
        for i in range(n_runs):
            rng = np.random.default_rng(70000 + i)
            truth, meas = simulate_radar_scenario(
                dt=DT, n_steps=n_steps, q_psd=q_psd, sigma_range=sr,
                sigma_bearing=sb, x0=x0, rng=rng,
            )
            if min_range is None:
                min_range = float(np.min(np.hypot(truth[:, 0], truth[:, 2])))
            x_init = np.array([truth[0, 0] + 10.0, 0.0, truth[0, 2] + 10.0, 0.0])
            ekf, ukf = _make_filters(f, q, r, x_init, p0)
            flt = ekf if name == "EKF" else ukf
            res = flt.run(meas)
            err = truth - res.x_post
            rmses.append(float(np.sqrt(np.mean(err[:, [0, 2]] ** 2))))
            nv = nees(err[10:], res.p_post[10:])
            neeses.append(float(np.mean(nv)))
            if nv[-1] > thresh:
                diverged += 1
        stats_out[name] = {
            "mean_rmse": float(np.mean(rmses)),
            "median_rmse": float(np.median(rmses)),
            "p90_rmse": float(np.percentile(rmses, 90)),
            "mean_nees": float(np.mean(neeses)),
            "diverged": float(diverged),
        }
    print(f"  closest approach of the truth to the sensor: {min_range:.1f} m")
    print(f"  sigma_bearing {sb} rad => cross-range error at closest approach "
          f"~{min_range * sb:.0f} m, comparable with the range itself")
    print(f"  {n_runs} independent runs x {n_steps} steps; divergence threshold "
          f"chi2_4(0.9999) = {thresh:.2f}")
    print()
    hdr = (
        f"  {'filter':<8}{'mean RMSE':>12}{'median':>12}"
        f"{'p90':>12}{'mean NEES':>12}{'diverged':>10}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, s in stats_out.items():
        print(
            f"  {name:<8}{s['mean_rmse']:>12.3f}{s['median_rmse']:>12.3f}"
            f"{s['p90_rmse']:>12.3f}{s['mean_nees']:>12.3f}{int(s['diverged']):>10d}"
        )
    ok = (
        stats_out["UKF"]["mean_rmse"] < stats_out["EKF"]["mean_rmse"]
        and stats_out["UKF"]["mean_nees"] < stats_out["EKF"]["mean_nees"]
    )
    print()
    print(
        f"  RMSE ratio EKF/UKF = "
        f"{stats_out['EKF']['mean_rmse'] / stats_out['UKF']['mean_rmse']:.3f}, "
        f"NEES ratio EKF/UKF = "
        f"{stats_out['EKF']['mean_nees'] / stats_out['UKF']['mean_nees']:.3f}"
    )
    print("  NOTE: both filters are inconsistent here (mean NEES >> 4). The claim under")
    print("  test is comparative degradation, not consistency; neither filter should be")
    print("  trusted in this regime.")
    print(f"  verdict: {'PASS' if ok else 'FAIL'} (UKF lower on both RMSE and NEES)")
    return ok


def part_c() -> bool:
    print()
    print("=" * 78)
    print("PART C - control: on a LINEAR measurement both reduce to the linear KF")
    print("=" * 78)
    f, q = constant_velocity_2d(DT, 0.05)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    r = np.diag([9.0, 9.0])
    p0 = np.diag([100.0, 10.0, 100.0, 10.0])
    x_init = np.zeros(4)
    rng = np.random.default_rng(2718)
    _, meas = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0, 0.0, -1.0]), 150, rng)
    kf = KalmanFilter(f, h, q, r, x_init, p0)
    ekf = ExtendedKalmanFilter(
        lambda x: f @ x, lambda x: h @ x, q, r, x_init, p0,
        f_jac=lambda x: f, h_jac=lambda x: h,
    )
    ukf = UnscentedKalmanFilter(
        lambda x: f @ x, lambda x: h @ x, q, r, x_init, p0, alpha=1.0, beta=2.0, kappa=0.0
    )
    rk, re, ru = kf.run(meas), ekf.run(meas), ukf.run(meas)
    scale_x = float(np.max(np.abs(rk.x_post)))
    scale_p = float(np.max(np.abs(rk.p_post)))
    d_ekf_x = float(np.max(np.abs(rk.x_post - re.x_post))) / scale_x
    d_ekf_p = float(np.max(np.abs(rk.p_post - re.p_post))) / scale_p
    d_ukf_x = float(np.max(np.abs(rk.x_post - ru.x_post))) / scale_x
    d_ukf_p = float(np.max(np.abs(rk.p_post - ru.p_post))) / scale_p
    print(f"  EKF vs KF: max relative |dx| = {d_ekf_x:.3e}, |dP| = {d_ekf_p:.3e}")
    print(f"  UKF vs KF: max relative |dx| = {d_ukf_x:.3e}, |dP| = {d_ukf_p:.3e}")
    ok = max(d_ekf_x, d_ekf_p, d_ukf_x, d_ukf_p) < 1e-11
    print(f"  verdict (tolerance 1e-11 relative): {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = [part_a(), part_b(), part_c()]
    print()
    print("=" * 78)
    print(f"OVERALL v3: {'PASS' if all(results) else 'FAIL'}")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
