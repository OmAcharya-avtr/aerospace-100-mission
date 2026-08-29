"""V3 — UKF matches EKF when the problem is near-linear and degrades better when it is not.

Run from ``products/P012/``::

    PYTHONPATH=src python validation/v3_ukf_vs_ekf.py

Three cases, in order of increasing nonlinearity.

**V3a — exactly linear.** Both nonlinear filters are given a linear ``f`` and
``h``. The unscented transform is exact for affine maps and the EKF Jacobian is
exact, so both must reproduce the linear Kalman filter to machine precision.
This is the strongest available known-answer test for the two nonlinear filters.

**V3b — near-linear.** A 2-D constant-velocity target observed in range and
bearing from a sensor 50 km away. What controls the nonlinearity is the ratio
of the *filter's own* position uncertainty to the range: the measured
steady-state position RMSE is ≈ 19.5 m, so ``σ/r ≈ 3.9×10⁻⁴`` and the
second-order term in the bearing expansion is ``O(1.5×10⁻⁷)`` rad — four
orders below the 1 mrad bearing noise. The two filters should therefore agree
to well within a percent.

**V3c — strongly nonlinear.** The Kitagawa (1987) univariate non-stationary
growth model with the standard ``Q = 10``, ``R = 1``. The measurement ``z =
x²/20`` is even, so the exact posterior is bimodal, and ``∂h/∂x = x/10``
vanishes at ``x = 0`` — the EKF gain collapses there. This is where the
"degrades more gracefully" claim is either supported by numbers or not.

References: Julier & Uhlmann (2004), *Proceedings of the IEEE* **92**(3),
401–422; Wan & van der Merwe (2000), *IEEE AS-SPCC*, 153–158; Kitagawa (1987),
*JASA* **82**(400), 1032–1041; Gordon, Salmond & Smith (1993), *IEE
Proceedings-F* **140**(2), 107–113.
"""

from __future__ import annotations

import numpy as np

from navbench.consistency import assess
from navbench.ekf import ExtendedKalmanFilter
from navbench.kf import KalmanFilter
from navbench.models import ConstantVelocity, RangeBearing, UnivariateGrowth
from navbench.ukf import SigmaPointSpec, UnscentedKalmanFilter


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def linear_equivalence() -> bool:
    """Exactly linear problem: EKF and UKF must reproduce the linear KF."""
    model = ConstantVelocity(dt=1.0, q_psd=0.1, sigma_pos=5.0, dim=2)
    f, q, h, r = model.f(), model.q(1.0), model.h(), model.r()
    rng = np.random.default_rng(11)
    x0 = np.array([0.0, 10.0, 0.0, -5.0])
    _, zs = model.simulate(x0, 120, rng)
    p0 = np.diag([100.0, 25.0] * 2)

    kf = KalmanFilter(f=f, q=q, h=h, r=r, x=x0.copy(), p=p0.copy())
    ekf = ExtendedKalmanFilter(
        f=lambda x, dt: f @ x, h=lambda x: h @ x, q=q, r=r, x=x0.copy(), p=p0.copy(),
        f_jac=lambda x, dt: f, h_jac=lambda x: h,
    )
    ukf = UnscentedKalmanFilter(
        f=lambda x, dt: f @ x, h=lambda x: h @ x, q=q, r=r, x=x0.copy(), p=p0.copy(),
        spec=SigmaPointSpec(alpha=1.0, beta=2.0, kappa=0.0),
    )
    dx_ekf = dx_ukf = dp_ekf = dp_ukf = 0.0
    for k in range(120):
        for filt in (kf, ekf, ukf):
            filt.predict()
            filt.update(zs[k])
        dx_ekf = max(dx_ekf, float(np.max(np.abs(ekf.x - kf.x))))
        dx_ukf = max(dx_ukf, float(np.max(np.abs(ukf.x - kf.x))))
        dp_ekf = max(dp_ekf, float(np.max(np.abs(ekf.p - kf.p))))
        dp_ukf = max(dp_ukf, float(np.max(np.abs(ukf.p - kf.p))))
    scale = float(np.max(np.abs(kf.p)))
    print(f"max |x_EKF - x_KF| over 120 steps = {dx_ekf:.3e} m")
    print(f"max |x_UKF - x_KF| over 120 steps = {dx_ukf:.3e} m")
    print(f"max |P_EKF - P_KF| over 120 steps = {dp_ekf:.3e}  (relative {dp_ekf / scale:.3e})")
    print(f"max |P_UKF - P_KF| over 120 steps = {dp_ukf:.3e}  (relative {dp_ukf / scale:.3e})")
    ok = dx_ekf < 1e-9 and dx_ukf < 1e-8 and dp_ekf < 1e-9 and dp_ukf < 1e-8
    print("tolerances: EKF 1e-9, UKF 1e-8 (Cholesky round-off in the sigma points)")
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def _run_range_bearing(sensor: tuple[float, float], n_runs: int, seed: int, n_steps: int = 100):
    model = ConstantVelocity(dt=1.0, q_psd=0.05, sigma_pos=1.0, dim=2)
    meas = RangeBearing(sensor=sensor, sigma_range=20.0, sigma_bearing=1.0e-3)
    f, q = model.f(), model.q(1.0)
    p0 = np.diag([100.0, 25.0] * 2)
    out = {}
    for name in ("ekf", "ukf"):
        nees = np.zeros((n_runs, n_steps))
        nis = np.zeros((n_runs, n_steps))
        errs = np.zeros((n_runs, n_steps, 4))
        for i in range(n_runs):
            rng = np.random.default_rng(seed + 7919 * i)
            x0 = np.array([0.0, 30.0, 0.0, 20.0])
            xs, _ = model.simulate(x0, n_steps, rng)
            zs = meas.simulate(xs[1:], rng)
            xhat0 = x0 + np.linalg.cholesky(p0) @ rng.standard_normal(4)
            if name == "ekf":
                filt = ExtendedKalmanFilter(
                    f=lambda x, dt: f @ x, h=meas.h, q=q, r=meas.r(), x=xhat0, p=p0.copy(),
                    f_jac=lambda x, dt: f, h_jac=meas.h_jac,
                )
            else:
                filt = UnscentedKalmanFilter(
                    f=lambda x, dt: f @ x, h=meas.h, q=q, r=meas.r(), x=xhat0, p=p0.copy(),
                    spec=SigmaPointSpec(alpha=1.0, beta=2.0, kappa=0.0),
                )
            for k in range(n_steps):
                filt.predict()
                info = filt.update(zs[k])
                e = xs[k + 1] - filt.x
                errs[i, k] = e
                nees[i, k] = float(e @ np.linalg.solve(filt.p, e))
                nis[i, k] = info.nis
        out[name] = (errs, nees, nis)
    return out


def near_linear() -> bool:
    """Range/bearing at 50 km: the two filters must agree closely."""
    n_runs = 60
    res = _run_range_bearing(sensor=(-50_000.0, 0.0), n_runs=n_runs, seed=777)
    print(f"{'filter':>8s} {'pos RMSE [m]':>14s} {'ANEES':>8s} {'ANIS':>8s} "
          f"{'NEES CI':>22s} {'pass':>6s}")
    stats = {}
    for name in ("ekf", "ukf"):
        errs, nees, nis = res[name]
        rmse = float(np.sqrt(np.mean(errs[..., [0, 2]] ** 2) * 2))
        rep = assess(nees, dof=4, label=f"NEES[{name}]")
        irep = assess(nis, dof=2, label=f"NIS[{name}]")
        stats[name] = (rmse, rep.normalised_mean, irep.normalised_mean, rep.passed)
        print(f"{name:>8s} {rmse:14.5f} {rep.normalised_mean:8.4f} {irep.normalised_mean:8.4f} "
              f"[{rep.mean_ci[0]:9.4f},{rep.mean_ci[1]:9.4f}] "
              f"{'yes' if rep.passed else 'NO':>6s}")
    rel = abs(stats["ukf"][0] - stats["ekf"][0]) / stats["ekf"][0]
    print(f"\nrelative RMSE difference |UKF-EKF|/EKF = {rel * 100:.4f} %  "
          f"(claim: < 1 % on a near-linear problem)")
    print(f"nonlinearity parameter (position RMSE)/range = {stats['ekf'][0] / 50_000.0:.3e}")
    ok = rel < 0.01
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def strongly_nonlinear() -> bool:
    """Kitagawa growth model: measure how each filter fails."""
    model = UnivariateGrowth(q=10.0, r=1.0)
    n_runs, n_steps = 200, 100
    p0 = np.array([[5.0]])
    summary = {}
    for name in ("ekf", "ukf"):
        rmse_runs = np.zeros(n_runs)
        nees_all = np.zeros((n_runs, n_steps))
        diverged = 0
        for i in range(n_runs):
            rng = np.random.default_rng(31337 + 7919 * i)
            xs, zs = model.simulate(0.1, n_steps, rng)
            x0 = np.array([0.1]) + np.sqrt(p0[0, 0]) * rng.standard_normal(1)
            if name == "ekf":
                filt = ExtendedKalmanFilter(
                    f=lambda x, dt, kk=[0]: model.f(x, kk[0]), h=model.h,
                    q=np.array([[model.q]]), r=np.array([[model.r]]), x=x0, p=p0.copy(),
                    f_jac=lambda x, dt, kk=[0]: model.f_jac(x, kk[0]), h_jac=model.h_jac,
                )
            else:
                filt = UnscentedKalmanFilter(
                    f=lambda x, dt, kk=[0]: model.f(x, kk[0]), h=model.h,
                    q=np.array([[model.q]]), r=np.array([[model.r]]), x=x0, p=p0.copy(),
                    spec=SigmaPointSpec(alpha=1.0, beta=2.0, kappa=2.0),
                )
            errs = np.zeros(n_steps)
            for k in range(n_steps):
                filt.f = (lambda kk: (lambda x, dt: model.f(x, kk)))(k)
                if name == "ekf":
                    filt.f_jac = (lambda kk: (lambda x, dt: model.f_jac(x, kk)))(k)
                filt.predict()
                filt.update(zs[k])
                e = float(xs[k + 1, 0] - filt.x[0])
                errs[k] = e
                nees_all[i, k] = e * e / float(filt.p[0, 0])
            rmse_runs[i] = float(np.sqrt(np.mean(errs ** 2)))
            if rmse_runs[i] > 20.0:
                diverged += 1
        rep = assess(nees_all, dof=1, label=f"NEES[{name}]")
        summary[name] = (
            float(np.mean(rmse_runs)), float(np.median(rmse_runs)),
            float(np.percentile(rmse_runs, 90)), diverged, rep.normalised_mean,
        )
        print(f"{name:>4s}: RMSE mean {summary[name][0]:8.4f}  median {summary[name][1]:8.4f}  "
              f"p90 {summary[name][2]:8.4f}  runs with RMSE>20: {diverged:3d}/{n_runs}  "
              f"ANEES {summary[name][4]:9.4f}")
        print(f"      {rep.summary()}")
    e_rmse, u_rmse = summary["ekf"][0], summary["ukf"][0]
    e_bad, u_bad = summary["ekf"][3], summary["ukf"][3]
    print(f"\nUKF/EKF mean RMSE ratio      = {u_rmse / e_rmse:.4f} "
          f"({'UKF better' if u_rmse < e_rmse else 'EKF better'})")
    print(f"UKF/EKF gross-failure counts = {u_bad} vs {e_bad}")
    print("NOTE: neither filter is consistent here. The exact posterior of this model is")
    print("      bimodal; a Gaussian-assumed-density filter cannot represent it. The claim")
    print("      under test is only the RELATIVE degradation, not adequacy of either filter.")
    ok = u_rmse < e_rmse and u_bad <= e_bad
    print(f"RESULT: {'PASS' if ok else 'FAILED'}  "
          "(criterion: UKF has both lower mean RMSE and no more gross failures)")
    return ok


def main() -> int:
    """Run V3 and return a process exit code."""
    np.set_printoptions(linewidth=140)
    _rule("V3a — exactly linear problem: EKF and UKF must equal the linear KF")
    ok_a = linear_equivalence()
    _rule("V3b — near-linear range/bearing at 50 km: UKF matches EKF")
    ok_b = near_linear()
    _rule("V3c — strongly nonlinear Kitagawa growth model: graceful degradation")
    ok_c = strongly_nonlinear()
    _rule("V3 SUMMARY")
    for name, ok in (
        ("V3a linear equivalence", ok_a),
        ("V3b near-linear agreement", ok_b),
        ("V3c nonlinear degradation", ok_c),
    ):
        print(f"{name:<32s}: {'PASS' if ok else 'FAILED'}")
    return 0 if all((ok_a, ok_b, ok_c)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
