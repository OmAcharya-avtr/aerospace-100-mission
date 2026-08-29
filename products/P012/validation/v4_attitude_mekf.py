"""V4 — quaternion algebra, MEKF reset behaviour and attitude-filter consistency.

Run from ``products/P012/``::

    PYTHONPATH=src python validation/v4_attitude_mekf.py

Checks
------
**V4a — algebra against an independent implementation.** Every rotation
operation is cross-checked against ``scipy.spatial.transform.Rotation`` (a
completely separate implementation, scalar-LAST storage) and against
hand-computable 90° rotations. This verifies the Hamilton/active/scalar-first
convention end to end, including the composition order ``R(p⊗q) = R(p)R(q)``.

**V4b — normalisation and the exponential/log maps.** Round-trip
``rotvec → quat → rotvec``, unit-norm preservation under composition, the
small-angle limit of ``δq(a) ≈ [1, a/2]`` and the size of its truncation error.

**V4c — MEKF reset.** The defining property of the multiplicative reset: after
``q̂ ← q̂ ⊗ δq(a⁺)`` the rotation actually applied must equal the estimated
error rotation, the reference quaternion must stay unit-norm to machine
precision over thousands of updates, and the error state must be exactly zero
afterwards (it is not stored, it is folded in). Markley's second-order
covariance reset is also exercised and its magnitude reported.

**V4d — error-state transition against finite differences.** The 6×6 ``Φ`` and
the measurement Jacobians are checked against numerical differentiation of the
true nonlinear maps — this is what catches a sign error in the error
definition, the classic MEKF bug.

**V4e — attitude-filter consistency.** A gyro + star-tracker MEKF is Monte
Carlo'd and its 6-state NEES tested against the chi-squared bounds.

**V4f — integration order.** Rectangular versus trapezoidal gyro integration on
the same trajectory, quantifying the deterministic drift that no ``Q`` can
absorb.

References: Markley & Crassidis (2014) §2.9, §6.2; Lefferts, Markley & Shuster
(1982), *JGCD* **5**(5), 417–429; Markley (2003), *JGCD* **26**(2), 311–317;
Shepperd (1978), *J. Guidance and Control* **1**(3), 223–224.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from navbench.consistency import assess
from navbench.mekf import MekfConfig, MultiplicativeEKF, error_state_transition
from navbench.quaternion import (
    attitude_matrix,
    dcm_to_quat,
    quat_angle_between,
    quat_canonical,
    quat_conjugate,
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_dcm,
    quat_to_rotvec,
    skew,
    small_angle_quat,
)
from navbench.sensors import GyroParams, StarTrackerParams, simulate_gyro, simulate_star_tracker
from navbench.truth import generate_attitude


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _to_scipy(q: np.ndarray) -> Rotation:
    """navbench scalar-first -> scipy scalar-last."""
    return Rotation.from_quat(np.roll(np.asarray(q, dtype=float), -1))


def algebra() -> bool:
    """Cross-check against scipy and against hand-computed 90-degree rotations."""
    rng = np.random.default_rng(2026)
    ok = True

    # Hand check: 90 deg about +z maps x -> y  (active rotation).
    qz = quat_from_rotvec([0.0, 0.0, np.pi / 2.0])
    v = quat_rotate(qz, [1.0, 0.0, 0.0])
    hand_err = float(np.max(np.abs(v - np.array([0.0, 1.0, 0.0]))))
    print(f"R(90 deg about +z) @ x_hat        = {np.array2string(v, precision=15)}")
    print(f"  hand-expected [0, 1, 0], max|err| = {hand_err:.3e}  (tol 1e-15)")
    ok = ok and hand_err < 1e-15

    # Hand check: composition order.
    qy = quat_from_rotvec([0.0, np.pi / 2.0, 0.0])
    comp = quat_to_dcm(quat_multiply(qz, qy))
    prod = quat_to_dcm(qz) @ quat_to_dcm(qy)
    comp_err = float(np.max(np.abs(comp - prod)))
    print(f"max|R(p (x) q) - R(p)R(q)|        = {comp_err:.3e}  (tol 1e-14)")
    ok = ok and comp_err < 1e-14

    # Random cross-checks against scipy.
    n = 500
    d_rot, d_dcm, d_rv, d_mul, d_inv = [], [], [], [], []
    for _ in range(n):
        a = quat_normalize(rng.standard_normal(4))
        b = quat_normalize(rng.standard_normal(4))
        u = rng.standard_normal(3)
        d_rot.append(np.max(np.abs(quat_rotate(a, u) - _to_scipy(a).apply(u))))
        d_dcm.append(np.max(np.abs(quat_to_dcm(a) - _to_scipy(a).as_matrix())))
        d_rv.append(np.max(np.abs(quat_to_rotvec(a) - _to_scipy(a).as_rotvec())))
        d_mul.append(
            np.max(np.abs(
                quat_to_dcm(quat_multiply(a, b)) - (_to_scipy(a) * _to_scipy(b)).as_matrix()
            ))
        )
        rt = dcm_to_quat(quat_to_dcm(a))
        d_inv.append(quat_angle_between(rt, a))
    rows = [
        ("quat_rotate vs Rotation.apply", float(np.max(d_rot)), 1e-14),
        ("quat_to_dcm vs Rotation.as_matrix", float(np.max(d_dcm)), 1e-14),
        ("quat_to_rotvec vs as_rotvec", float(np.max(d_rv)), 1e-13),
        ("composition vs Rotation product", float(np.max(d_mul)), 1e-13),
        ("dcm_to_quat round trip [rad]", float(np.max(d_inv)), 1e-13),
    ]
    print(f"\n{'check (N=500 random rotations)':<40s} {'max deviation':>15s} {'tol':>10s} {'':>6s}")
    for name, val, tol in rows:
        good = val < tol
        ok = ok and good
        print(f"{name:<40s} {val:15.4e} {tol:10.1e} {'PASS' if good else 'FAIL':>6s}")

    # 180 degree case, where the naive trace formula for dcm_to_quat fails.
    q180 = quat_from_rotvec([0.0, np.pi, 0.0])
    err180 = quat_angle_between(dcm_to_quat(quat_to_dcm(q180)), q180)
    print(f"\n180 deg round trip through the DCM: angular error {err180:.3e} rad (tol 1e-12)")
    ok = ok and err180 < 1e-12

    # attitude_matrix is the transpose (inertial -> body).
    a = quat_normalize(rng.standard_normal(4))
    at_err = float(np.max(np.abs(attitude_matrix(a) - quat_to_dcm(a).T)))
    print(f"max|A(q) - R(q)^T| = {at_err:.3e}")
    ok = ok and at_err == 0.0
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def normalisation() -> bool:
    """Norm preservation and the small-angle truncation error."""
    rng = np.random.default_rng(7)
    q = quat_normalize(rng.standard_normal(4))
    worst = 0.0
    for _ in range(20_000):
        q = quat_multiply(q, quat_from_rotvec(1e-2 * rng.standard_normal(3)))
        worst = max(worst, abs(float(np.linalg.norm(q)) - 1.0))
    print(f"max |‖q‖ - 1| after 20000 un-renormalised compositions = {worst:.3e}")
    print("  (the Hamilton product of unit quaternions is unit by construction; this")
    print("   measures pure floating-point drift, and is why renormalising every step")
    print("   is cheap insurance rather than a necessity)")
    ok = worst < 1e-12

    print(f"\n{'|a| [rad]':>12s} {'angle(δq_1st, exp(a)) [rad]':>30s} {'|a|^3/48 bound':>18s}")
    for mag in (1e-4, 1e-3, 1e-2, 1e-1, 0.5):
        a = mag * np.array([1.0, 0.0, 0.0])
        err = quat_angle_between(small_angle_quat(a), quat_from_rotvec(a))
        bound = mag ** 3 / 48.0
        print(f"{mag:12.1e} {err:30.6e} {bound:18.6e}")
        ok = ok and err <= 1.05 * bound + 1e-15
    print("first-order error quaternion agrees with the exact exponential map to |a|^3/48")
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def reset_behaviour() -> bool:
    """MEKF reset: applied rotation equals the estimated error; norm stays 1."""
    rng = np.random.default_rng(99)
    ok = True
    p0 = np.diag([1e-6] * 3 + [1e-12] * 3)
    mekf = MultiplicativeEKF(q_ref=[1.0, 0.0, 0.0, 0.0], bias=np.zeros(3), p=p0.copy())
    r = np.eye(3) * (1e-5) ** 2
    q_before = mekf.q_ref.copy()
    b_before = mekf.bias.copy()
    q_meas = quat_normalize(quat_multiply(q_before, quat_from_rotvec([3e-4, -2e-4, 1e-4])))
    info = mekf.update_quaternion(q_meas, r)
    dx = info.gain @ info.innovation
    applied = quat_to_rotvec(quat_multiply(quat_conjugate(q_before), mekf.q_ref))
    err_rot = float(np.max(np.abs(applied - dx[:3])))
    err_bias = float(np.max(np.abs((mekf.bias - b_before) - dx[3:])))
    print(f"estimated attitude error a+           = {np.array2string(dx[:3], precision=9)}")
    print(f"rotation actually applied to q_ref    = {np.array2string(applied, precision=9)}")
    print(f"max|applied - a+|                     = {err_rot:.3e}  (tol 1e-12; the exact")
    print("   exponential map is used, so the two agree to round-off)")
    print(f"max|Δb - δb+|                         = {err_bias:.3e}  (tol 1e-18)")
    print(f"resets performed                      = {mekf.n_resets}")
    ok = ok and err_rot < 1e-12 and err_bias < 1e-18 and mekf.n_resets == 1

    # Norm preservation over many propagate/update cycles.
    mekf2 = MultiplicativeEKF(q_ref=quat_normalize(rng.standard_normal(4)), bias=np.zeros(3),
                              p=p0.copy())
    worst = 0.0
    for _ in range(5000):
        mekf2.propagate(np.array([0.01, -0.02, 0.005]), 0.1)
        mekf2.update_quaternion(
            quat_normalize(quat_multiply(mekf2.q_ref, quat_from_rotvec(1e-5 * rng.standard_normal(3)))),
            r,
        )
        worst = max(worst, abs(float(np.linalg.norm(mekf2.q_ref)) - 1.0))
    print(f"\nmax |‖q_ref‖ - 1| over 5000 propagate+update cycles = {worst:.3e}  (tol 1e-14)")
    ok = ok and worst < 1e-14

    # Second-order covariance reset magnitude.
    mekf3 = MultiplicativeEKF(q_ref=[1.0, 0, 0, 0], bias=np.zeros(3), p=p0.copy(),
                              config=MekfConfig(covariance_reset=True))
    mekf4 = MultiplicativeEKF(q_ref=[1.0, 0, 0, 0], bias=np.zeros(3), p=p0.copy(),
                              config=MekfConfig(covariance_reset=False))
    for m in (mekf3, mekf4):
        m.update_quaternion(q_meas, r)
    rel = float(np.max(np.abs(mekf3.p - mekf4.p)) / np.max(np.abs(mekf4.p)))
    print(f"relative effect of Markley's second-order covariance reset = {rel:.3e}")
    print("   (O(|a+|^2); negligible at star-tracker accuracies, as documented)")
    ok = ok and rel < 1e-3

    # Sign double cover: a measurement of -q must give the same update as +q.
    m5 = MultiplicativeEKF(q_ref=[1.0, 0, 0, 0], bias=np.zeros(3), p=p0.copy())
    m6 = MultiplicativeEKF(q_ref=[1.0, 0, 0, 0], bias=np.zeros(3), p=p0.copy())
    m5.update_quaternion(q_meas, r)
    m6.update_quaternion(-q_meas, r)
    dbl = quat_angle_between(m5.q_ref, m6.q_ref)
    print(f"attitude difference between updates with q_meas and -q_meas = {dbl:.3e} rad "
          "(tol 1e-15)")
    ok = ok and dbl < 1e-15
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def jacobians() -> bool:
    """Error-state transition and measurement Jacobians vs finite differences."""
    rng = np.random.default_rng(4)
    ok = True

    # Phi: propagate a perturbed truth alongside the estimate and compare.
    q_hat = quat_normalize(rng.standard_normal(4))
    b_hat = np.array([1e-4, -2e-4, 3e-4])
    a0 = np.array([1e-4, -3e-4, 2e-4])
    db0 = np.array([2e-5, 1e-5, -3e-5])
    q_true = quat_normalize(quat_multiply(q_hat, quat_from_rotvec(a0)))
    w_meas = np.array([0.05, -0.03, 0.02])
    print(f"{'dt [s]':>8s} {'|a_actual|':>14s} {'|a_pred - a_actual|':>22s} {'relative':>12s}")
    for dt in (0.01, 0.1, 1.0):
        w_hat = w_meas - b_hat
        w_true = w_meas - (b_hat + db0)
        q_hat2 = quat_normalize(quat_multiply(q_hat, quat_from_rotvec(w_hat * dt)))
        q_true2 = quat_normalize(quat_multiply(q_true, quat_from_rotvec(w_true * dt)))
        a_actual = quat_to_rotvec(quat_multiply(quat_conjugate(q_hat2), q_true2))
        pred = error_state_transition(w_hat, dt) @ np.concatenate((a0, db0))
        d = float(np.linalg.norm(pred[:3] - a_actual))
        rel = d / float(np.linalg.norm(a_actual))
        print(f"{dt:8.3g} {np.linalg.norm(a_actual):14.6e} {d:22.6e} {rel:12.3e}")
        ok = ok and rel < 5e-5

    # H for the vector measurement: dz/da by central differences.
    ref = np.array([0.6, -0.8, 0.0])
    ref = ref / np.linalg.norm(ref)
    b_pred = attitude_matrix(q_hat) @ ref
    h_analytic = skew(b_pred)
    h_numeric = np.zeros((3, 3))
    eps = 1e-7
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        qp = quat_multiply(q_hat, quat_from_rotvec(e))
        qm = quat_multiply(q_hat, quat_from_rotvec(-e))
        h_numeric[:, j] = (attitude_matrix(qp) @ ref - attitude_matrix(qm) @ ref) / (2.0 * eps)
    d = float(np.max(np.abs(h_analytic - h_numeric)))
    print(f"\nvector-measurement Jacobian: max|[b_hat x] - numerical dz/da| = {d:.3e} (tol 1e-7)")
    ok = ok and d < 1e-7

    # H for the quaternion measurement should be [I 0].
    z_num = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        qp = quat_multiply(q_hat, quat_from_rotvec(e))
        dq = quat_multiply(quat_conjugate(q_hat), qp)
        z_num[:, j] = 2.0 * dq[1:] / eps
    d2 = float(np.max(np.abs(z_num - np.eye(3))))
    print(f"quaternion-measurement Jacobian: max|H - I| = {d2:.3e} (tol 1e-7)")
    ok = ok and d2 < 1e-7
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def integration_order() -> bool:
    """Rectangular vs trapezoidal gyro integration: the drift that Q cannot absorb."""
    inertia = np.diag([10.0, 7.0, 5.0])
    print(f"{'dt [s]':>8s} {'rectangular [arcsec]':>22s} {'trapezoidal [arcsec]':>22s} "
          f"{'ratio':>10s}")
    ok = True
    for dt in (0.1, 0.05, 0.02):
        n = int(200.0 / dt)
        tr = generate_attitude(
            inertia=inertia, q0=[1.0, 0, 0, 0], omega0=[0.02, -0.01, 0.03], dt=dt, n_steps=n
        )
        q_rect = tr.q[0].copy()
        q_trap = tr.q[0].copy()
        for k in range(n):
            q_rect = quat_normalize(quat_multiply(q_rect, quat_from_rotvec(tr.omega[k] * dt)))
            w_mid = 0.5 * (tr.omega[k] + tr.omega[k + 1])
            q_trap = quat_normalize(quat_multiply(q_trap, quat_from_rotvec(w_mid * dt)))
        er = np.degrees(quat_angle_between(q_rect, tr.q[n])) * 3600.0
        et = np.degrees(quat_angle_between(q_trap, tr.q[n])) * 3600.0
        print(f"{dt:8.3g} {er:22.4f} {et:22.6f} {er / et:10.1f}")
        ok = ok and et < er
    print("Rectangular error scales as O(dt) after 200 s (a coherent, non-white drift);")
    print("trapezoidal as O(dt^2). This is why MultiplicativeEKF.propagate accepts")
    print("omega_prev, and why the consistency test below uses it.")
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def mekf_consistency(n_runs: int = 30) -> bool:
    """Gyro + star tracker MEKF: 6-state NEES against the chi-squared bounds."""
    inertia = np.diag([10.0, 7.0, 5.0])
    dt = 0.1
    n_steps = 2000
    burn = 500
    sigma_a0, sigma_b0 = 1.0e-3, 1.0e-6
    arw, rrw = 3.0e-5, 1.0e-8
    p0 = np.diag([sigma_a0 ** 2] * 3 + [sigma_b0 ** 2] * 3)
    stp = StarTrackerParams(sigma_cross=1.0e-5, sigma_boresight=7.0e-5, decimation=10)
    r = stp.noise_covariance()
    nees = np.zeros((n_runs, n_steps - burn))
    att_err = np.zeros((n_runs, n_steps - burn))
    bias_err = np.zeros(n_runs)
    for i in range(n_runs):
        rng = np.random.default_rng(9000 + i)
        truth = generate_attitude(
            inertia=inertia, q0=[1.0, 0, 0, 0], omega0=[0.02, -0.01, 0.03],
            dt=dt, n_steps=n_steps,
        )
        b0 = sigma_b0 * rng.standard_normal(3)
        gyro = simulate_gyro(
            truth, GyroParams(arw=arw, bias_sigma=0.0, bias_tau=np.inf, rrw=rrw, initial_bias=b0),
            rng=rng,
        )
        idx, qm = simulate_star_tracker(truth, stp, rng=rng)
        lookup = {int(v): j for j, v in enumerate(idx)}
        q_hat0 = quat_normalize(
            quat_multiply(truth.q[0], quat_from_rotvec(sigma_a0 * rng.standard_normal(3)))
        )
        mekf = MultiplicativeEKF(
            q_ref=q_hat0, bias=np.zeros(3), p=p0.copy(),
            config=MekfConfig(sigma_v=arw, sigma_u=rrw),
        )
        for k in range(1, n_steps):
            mekf.propagate(gyro.omega[k], dt, omega_prev=gyro.omega[k - 1])
            if k in lookup:
                mekf.update_quaternion(qm[lookup[k]], r)
            if k >= burn:
                e6 = mekf.error_state(truth.q[k], gyro.bias[k])
                nees[i, k - burn] = float(e6 @ np.linalg.solve(mekf.p, e6))
                att_err[i, k - burn] = quat_angle_between(mekf.q_ref, truth.q[k])
        bias_err[i] = float(np.linalg.norm(mekf.bias - gyro.bias[-1]))
    rep = assess(nees, dof=6, label="NEES[MEKF]")
    print(f"Monte Carlo: M = {n_runs} runs x {n_steps - burn} scored steps, dt = {dt} s, "
          f"star tracker every {stp.decimation * dt:g} s")
    print(f"gyro ARW = {arw:.1e} rad/sqrt(s), RRW = {rrw:.1e} rad/s^1.5; tracker "
          f"sigma_cross = {stp.sigma_cross:.1e} rad, sigma_bore = {stp.sigma_boresight:.1e} rad")
    print(f"\nRMS attitude error = {np.degrees(np.sqrt(np.mean(att_err ** 2))) * 3600:.3f} arcsec")
    print(f"RMS final bias error = {np.sqrt(np.mean(bias_err ** 2)):.4e} rad/s "
          f"({np.degrees(np.sqrt(np.mean(bias_err ** 2))) * 3600:.4f} arcsec/s)")
    print(rep.summary())
    ok = rep.passed
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def main() -> int:
    """Run V4 and return a process exit code."""
    np.set_printoptions(linewidth=140)
    _rule("V4a — quaternion algebra vs scipy.spatial.transform.Rotation and hand cases")
    ok_a = algebra()
    _rule("V4b — normalisation and small-angle error quaternion")
    ok_b = normalisation()
    _rule("V4c — MEKF reset behaviour")
    ok_c = reset_behaviour()
    _rule("V4d — error-state transition and measurement Jacobians vs finite differences")
    ok_d = jacobians()
    _rule("V4e — gyro integration order")
    ok_e = integration_order()
    _rule("V4f — MEKF consistency (NEES) with gyro + star tracker")
    ok_f = mekf_consistency()
    _rule("V4 SUMMARY")
    for name, ok in (
        ("V4a algebra vs scipy", ok_a),
        ("V4b normalisation / small angle", ok_b),
        ("V4c MEKF reset", ok_c),
        ("V4d Jacobians", ok_d),
        ("V4e integration order", ok_e),
        ("V4f MEKF NEES consistency", ok_f),
    ):
        print(f"{name:<36s}: {'PASS' if ok else 'FAILED'}")
    return 0 if all((ok_a, ok_b, ok_c, ok_d, ok_e, ok_f)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
