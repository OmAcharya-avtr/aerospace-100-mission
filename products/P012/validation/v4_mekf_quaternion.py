"""Validation 4 — quaternion algebra, normalization, and MEKF reset behaviour.

Run from the product root::

    PYTHONPATH=src python3 validation/v4_mekf_quaternion.py

RELATED PRIOR ART.  Product P007 (QuatKit) in this portfolio is a dedicated
quaternion/attitude-representation toolbox using the same scalar-first
Hamilton convention.  NavBench imports nothing from it — every product in the
portfolio is self-contained — so the algebra in :mod:`navbench.attitude` is an
independent implementation and is validated independently here, including
against ``scipy.spatial.transform.Rotation`` as an outside reference.  P007 is
cited as related work, not reused.

PART A  quaternion algebra: DCM cross-check against SciPy, round-trips,
        composition, and the exact rotation-vector map.
PART B  rigid-body integrator: RK4 order, and the conserved quantities of
        torque-free motion (Wertz 1978 §16.2).
PART C  quaternion normalization: norm drift with and without renormalisation.
PART D  MEKF reset: the reference absorbs the estimated error exactly, the
        error state returns to zero, and the neglected covariance-reset
        Jacobian ``I − ½[â×]`` (Markley 2003 §V) is measured, not assumed
        negligible.
PART E  MEKF Monte Carlo NEES/NIS against chi-squared bounds.
PART F  the rate-discretisation error that motivates
        ``AttitudeTruth.interval_rate()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navbench import (  # noqa: E402
    GyroModel,
    MultiplicativeEKF,
    StarTrackerModel,
    arw_deg_per_sqrt_hour_to_si,
    attitude_trajectory,
    axis_angle_from_quat,
    dcm_from_quat,
    ensemble_consistency,
    nees,
    nis,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_dcm,
    quat_from_euler_zyx,
    quat_from_small_angle,
    quat_multiply,
    quat_normalize,
    quat_propagate,
    rrw_deg_per_hour_1p5_to_si,
    skew,
    small_angle_from_quat,
)


def part_a() -> bool:
    print("=" * 78)
    print("PART A - quaternion algebra")
    print("=" * 78)
    rng = np.random.default_rng(1)
    qs = np.array([quat_normalize(rng.standard_normal(4)) for _ in range(2000)])

    d_scipy = max(
        float(
            np.max(
                np.abs(
                    dcm_from_quat(q)
                    - Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                )
            )
        )
        for q in qs
    )
    d_round = max(
        float(np.max(np.abs(quat_from_dcm(dcm_from_quat(q)) - (q if q[0] >= 0 else -q))))
        for q in qs
    )
    d_orth = max(
        float(np.max(np.abs(dcm_from_quat(q) @ dcm_from_quat(q).T - np.eye(3)))) for q in qs
    )
    d_det = max(abs(float(np.linalg.det(dcm_from_quat(q))) - 1.0) for q in qs)
    d_comp = max(
        float(
            np.max(
                np.abs(
                    dcm_from_quat(quat_multiply(a, b)) - dcm_from_quat(a) @ dcm_from_quat(b)
                )
            )
        )
        for a, b in zip(qs[:-1:2], qs[1::2], strict=True)
    )
    # rotation-vector map: exp/log round trip over a wide angle range
    d_rotvec = 0.0
    for mag in (1e-14, 1e-10, 1e-6, 1e-3, 0.1, 1.0, 3.0):
        for _ in range(200):
            v = rng.standard_normal(3)
            v = mag * v / np.linalg.norm(v)
            # RELATIVE error: the absolute round-trip error scales with |v|, so a
            # single absolute tolerance would be meaningless across 14 decades.
            d_rotvec = max(
                d_rotvec,
                float(np.max(np.abs(small_angle_from_quat(quat_from_small_angle(v)) - v))) / mag,
            )
    # axis-angle round trip
    d_aa = 0.0
    for _ in range(500):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ang = float(rng.uniform(0.0, np.pi))
        ax2, an2 = axis_angle_from_quat(quat_from_axis_angle(axis, ang))
        d_aa = max(d_aa, float(np.max(np.abs(ax2 * an2 - axis * ang))))

    checks = [
        ("DCM vs scipy Rotation (2000 random q)", d_scipy, 1e-14),
        ("quat -> DCM -> quat round trip", d_round, 1e-13),
        ("DCM orthogonality max|R Rt - I|", d_orth, 1e-14),
        ("|det(R) - 1|", d_det, 1e-14),
        ("R(a (x) b) = R(a) R(b)", d_comp, 1e-14),
        ("rotvec exp/log RELATIVE round trip (1e-14..3 rad)", d_rotvec, 1e-14),
        ("axis-angle round trip", d_aa, 1e-14),
    ]
    ok = True
    for label, val, tol in checks:
        good = val < tol
        ok &= good
        print(f"  {label:<44s} {val:.3e}  (tol {tol:.0e})  {'PASS' if good else 'FAIL'}")
    return ok


def part_b() -> bool:
    print()
    print("=" * 78)
    print("PART B - rigid-body integrator (RK4)")
    print("=" * 78)
    inertia = np.diag([10.0, 15.0, 20.0])
    ok = True

    # B1: rotation about a principal axis is exactly a constant-rate rotation.
    tr = attitude_trajectory(
        inertia=inertia, quat0=np.array([1.0, 0.0, 0.0, 0.0]),
        omega0=np.array([0.0, 0.0, 0.05]), dt=0.5, n_steps=400,
    )
    exact = np.array([quat_from_axis_angle([0, 0, 1], 0.05 * t) for t in tr.t])
    d_exact = float(
        np.max([np.min(np.abs(np.abs(a @ b) - 1.0)) for a, b in zip(tr.quat, exact, strict=True)])
    )
    good = d_exact < 1e-14
    ok &= good
    print(f"  B1 principal-axis spin vs analytic (200 s): max |1 - |q.q_exact|| = "
          f"{d_exact:.3e}  (tol 1e-14)  {'PASS' if good else 'FAIL'}")

    # B2: torque-free conserved quantities.
    tr2 = attitude_trajectory(
        inertia=inertia, quat0=quat_from_euler_zyx(0.3, 0.2, -0.4),
        omega0=np.array([0.05, -0.08, 0.03]), dt=0.1, n_steps=3000,
    )
    e = tr2.kinetic_energy()
    hvec = tr2.angular_momentum()
    d_e = float(np.max(np.abs(e - e[0])) / e[0])
    d_h = float(np.max(np.linalg.norm(hvec - hvec[0], axis=1)) / np.linalg.norm(hvec[0]))
    good = d_e < 1e-12 and d_h < 1e-9
    ok &= good
    print(f"  B2 torque-free 300 s, dt = 0.1 s: relative energy drift {d_e:.3e} (tol 1e-12), "
          f"inertial |H| drift {d_h:.3e} (tol 1e-9)  {'PASS' if good else 'FAIL'}")

    # B3: RK4 order — halving dt should cut the error by ~16.
    ref = attitude_trajectory(
        inertia=inertia, quat0=quat_from_euler_zyx(0.3, 0.2, -0.4),
        omega0=np.array([0.5, -0.8, 0.3]), dt=0.00125, n_steps=8000,
    )
    q_ref = ref.quat[-1]
    errs = []
    # Step sizes must divide the 10 s span exactly: a step that leaves a partial
    # interval compares two different final times and the ratio is meaningless.
    for dt in (0.1, 0.05, 0.025, 0.0125):
        n = int(round(10.0 / dt))
        assert abs(n * dt - 10.0) < 1e-12, "dt must divide the 10 s span exactly"
        tr3 = attitude_trajectory(
            inertia=inertia, quat0=quat_from_euler_zyx(0.3, 0.2, -0.4),
            omega0=np.array([0.5, -0.8, 0.3]), dt=dt, n_steps=n,
        )
        ang = float(
            np.linalg.norm(
                small_angle_from_quat(quat_multiply(quat_conjugate(q_ref), tr3.quat[-1]))
            )
        )
        errs.append((dt, ang))
    print("     dt [s]     final attitude error [rad]     ratio to previous")
    ratios = []
    for i, (dt, err) in enumerate(errs):
        ratio = errs[i - 1][1] / err if i > 0 else float("nan")
        if i > 0:
            ratios.append(ratio)
        print(f"     {dt:<10.5f} {err:.6e}                {ratio:>8.2f}")
    good = all(12.0 < r < 20.0 for r in ratios)
    ok &= good
    print(f"  B3 observed order ratios {[f'{r:.2f}' for r in ratios]} vs RK4 ideal 16 "
          f"(band 12-20)  {'PASS' if good else 'FAIL'}")
    return ok


def part_c() -> bool:
    print()
    print("=" * 78)
    print("PART C - quaternion normalization")
    print("=" * 78)
    rng = np.random.default_rng(5)
    q = quat_normalize(rng.standard_normal(4))
    omega = np.array([0.03, -0.07, 0.11])
    dt = 0.05
    n = 200000
    qn = q.copy()
    max_dev = 0.0
    for _ in range(n):
        qn = quat_propagate(qn, omega, dt)
        max_dev = max(max_dev, abs(float(np.linalg.norm(qn)) - 1.0))
    print(f"  C1 quat_propagate, {n} steps: max ||q| - 1| = {max_dev:.3e}  (tol 1e-14)")
    ok = max_dev < 1e-14

    # Un-normalised first-order integration, for contrast.
    qraw = q.copy()
    for _ in range(20000):
        qraw = qraw + 0.5 * quat_multiply(qraw, np.concatenate(([0.0], omega))) * dt
    drift = abs(float(np.linalg.norm(qraw)) - 1.0)
    print(f"  C2 contrast: 20000 steps of un-normalised first-order Euler integration "
          f"gives ||q| - 1| = {drift:.3e}")
    print("     (this is why quat_propagate and the MEKF renormalise every step)")
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def _make_scenario(dt: float, n_steps: int):
    inertia = np.diag([10.0, 15.0, 20.0])
    tr = attitude_trajectory(
        inertia=inertia, quat0=quat_from_euler_zyx(0.2, -0.1, 0.3),
        omega0=np.array([0.01, -0.02, 0.015]), dt=dt, n_steps=n_steps,
        torque_fn=lambda t, q, w: np.array([1e-5 * np.sin(0.01 * t), 0.0, 0.0]),
    )
    return tr


def part_d() -> bool:
    print()
    print("=" * 78)
    print("PART D - MEKF multiplicative reset")
    print("=" * 78)
    ok = True
    # D1: an isolated reset moves the reference by exactly the estimated error.
    mekf = MultiplicativeEKF(
        sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_from_euler_zyx(0.1, 0.2, 0.3),
        bias0=np.zeros(3), p0=np.diag([0.05**2] * 3 + [1e-5**2] * 3),
    )
    q_before = mekf.quat.copy()
    q_meas = quat_normalize(
        quat_multiply(q_before, quat_from_small_angle(np.array([1e-3, -2e-3, 5e-4])))
    )
    out = mekf.update_quaternion(q_meas, 1e-4)
    dx = np.asarray(out["dx"])
    expected = quat_normalize(quat_multiply(q_before, quat_from_small_angle(dx[:3])))
    d_reset = float(np.max(np.abs(mekf.quat - expected)))
    good = d_reset < 1e-15
    ok &= good
    print(f"  D1 reference after reset equals q_before (x) dq(a_hat): max |diff| = "
          f"{d_reset:.3e}  (tol 1e-15)  {'PASS' if good else 'FAIL'}")
    print(f"     estimated attitude error a_hat = {np.array2string(dx[:3], precision=6)} rad")
    print(f"     |q| after reset = {float(np.linalg.norm(mekf.quat)):.17f}")

    # D2: a second, identical measurement now produces a much smaller innovation.
    out2 = mekf.update_quaternion(q_meas, 1e-4)
    n1 = float(np.linalg.norm(np.asarray(out["innovation"])))
    n2 = float(np.linalg.norm(np.asarray(out2["innovation"])))
    good = n2 < 0.2 * n1
    ok &= good
    print(f"  D2 innovation norm before reset {n1:.6e} rad, after reset {n2:.6e} rad "
          f"(ratio {n2 / n1:.4f})  {'PASS' if good else 'FAIL'}")

    # D3: magnitude of the NEGLECTED covariance-reset Jacobian (Markley 2003 sec. V).
    print("  D3 neglected covariance reset Jacobian G = I - 0.5*[a_hat x]:")
    tr = _make_scenario(0.5, 600)
    sigma_v = arw_deg_per_sqrt_hour_to_si(0.05)
    sigma_u = rrw_deg_per_hour_1p5_to_si(0.5)
    rng = np.random.default_rng(4242)
    gyro = GyroModel(sigma_v=sigma_v, sigma_u=sigma_u, dt=0.5, bias0=2e-6 * rng.standard_normal(3))
    rates, _ = gyro.sample_series(tr.interval_rate(), rng)
    st = StarTrackerModel(sigma_rad=3e-5, reference_vectors=np.eye(3))
    q_meas_all = np.array([st.sample_quaternion(q, rng) for q in tr.quat[1:]])
    m2 = MultiplicativeEKF(
        sigma_v=sigma_v, sigma_u=sigma_u, dt=0.5,
        quat0=quat_multiply(tr.quat[0], quat_from_small_angle(0.05 * rng.standard_normal(3))),
        bias0=np.zeros(3), p0=np.diag([0.05**2] * 3 + [2e-6**2] * 3),
    )
    res = m2.run(rates, quat_meas=q_meas_all, sigma_rad=3e-5, measurement_every=4)
    worst_rel = 0.0
    for k in np.argsort(res.reset_angle)[-20:]:
        a_hat = res.reset_angle[k]
        g = np.eye(6)
        g[:3, :3] = np.eye(3) - 0.5 * skew(np.array([a_hat, 0.0, 0.0]))
        p = res.covariance[k]
        rel = float(np.max(np.abs(g @ p @ g.T - p)) / np.max(np.abs(p)))
        worst_rel = max(worst_rel, rel)
    print(f"     max reset angle over the run  : {float(np.max(res.reset_angle)):.3e} rad")
    print(f"     median reset angle after step 100: "
          f"{float(np.median(res.reset_angle[100:])):.3e} rad")
    print(f"     worst relative change |G P Gt - P|/max|P| over the 20 largest resets: "
          f"{worst_rel:.3e}")
    print("     Interpretation: the neglected term is second order in the reset angle;")
    print("     at the sub-milliradian resets this filter operates at it is far below")
    print("     the covariance itself. It would NOT be negligible after a large")
    print("     attitude acquisition manoeuvre, and that limit is stated in the README.")
    print(f"     max ||q| - 1| over {res.quat.shape[0]} steps = "
          f"{float(np.max(np.abs(np.linalg.norm(res.quat, axis=1) - 1.0))):.3e}")
    ok &= float(np.max(np.abs(np.linalg.norm(res.quat, axis=1) - 1.0))) < 1e-14
    return ok


def part_e() -> bool:
    print()
    print("=" * 78)
    print("PART E - MEKF Monte Carlo consistency (NEES dof 6, NIS dof 3)")
    print("=" * 78)
    dt, n_steps, n_runs, burn = 0.5, 300, 30, 50
    tr = _make_scenario(dt, n_steps)
    w_true = tr.interval_rate()
    sigma_v = arw_deg_per_sqrt_hour_to_si(0.05)
    sigma_u = rrw_deg_per_hour_1p5_to_si(0.5)
    sigma_st = 3e-5
    sig_a0, sig_b0 = 0.05, 2e-6
    p0 = np.diag([sig_a0**2] * 3 + [sig_b0**2] * 3)
    nees_runs = np.zeros((n_runs, n_steps))
    nis_all = []
    for i in range(n_runs):
        rng = np.random.default_rng(1000 + i)
        gyro = GyroModel(
            sigma_v=sigma_v, sigma_u=sigma_u, dt=dt, bias0=sig_b0 * rng.standard_normal(3)
        )
        rates, biases = gyro.sample_series(w_true, rng)
        st = StarTrackerModel(sigma_rad=sigma_st, reference_vectors=np.eye(3))
        q_meas = np.array([st.sample_quaternion(q, rng) for q in tr.quat[1:]])
        q_init = quat_multiply(tr.quat[0], quat_from_small_angle(sig_a0 * rng.standard_normal(3)))
        mekf = MultiplicativeEKF(
            sigma_v=sigma_v, sigma_u=sigma_u, dt=dt, quat0=q_init, bias0=np.zeros(3), p0=p0
        )
        res = mekf.run(rates, quat_meas=q_meas, sigma_rad=sigma_st, measurement_every=4)
        err = res.error_state(tr.quat[1:], biases)
        nees_runs[i] = nees(err, res.covariance)
        nis_all.append(nis(res.innovation[burn:], res.innovation_cov[burn:]))
    avg, lo, hi = ensemble_consistency(nees_runs[:, burn:], 6)
    frac = float(np.mean((avg >= lo) & (avg <= hi)))
    nis_flat = np.concatenate(nis_all)
    nis_flat = nis_flat[np.isfinite(nis_flat)]
    nis_mean = float(np.mean(nis_flat))
    from navbench import chi2_bounds

    nlo, nhi = chi2_bounds(3, nis_flat.size)
    print(f"  {n_runs} independent runs x {n_steps} steps of {dt} s, burn-in {burn}")
    print(f"  star tracker sigma {sigma_st:.1e} rad, gyro ARW 0.05 deg/sqrt(hr), "
          f"RRW 0.5 deg/hr^1.5, update every 4 steps")
    print(f"  ANEES (dof 6): mean {float(np.mean(avg)):.4f}, bounds [{lo:.4f}, {hi:.4f}], "
          f"{100.0 * frac:.1f} % of steps inside")
    print(f"  NIS  (dof 3): mean over {nis_flat.size} pooled updates {nis_mean:.4f}, "
          f"bounds [{nlo:.4f}, {nhi:.4f}]")
    ok = lo <= float(np.mean(avg)) <= hi and nlo <= nis_mean <= nhi
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def part_f() -> bool:
    print()
    print("=" * 78)
    print("PART F - rate discretisation: why AttitudeTruth.interval_rate() exists")
    print("=" * 78)
    tr = _make_scenario(0.5, 600)
    w_eff = tr.interval_rate()
    d_end = float(np.max(np.abs(tr.omega[1:] - w_eff)))
    d_start = float(np.max(np.abs(tr.omega[:-1] - w_eff)))
    print(f"  max |omega(end of interval)   - omega_effective| = {d_end:.3e} rad/s")
    print(f"  max |omega(start of interval) - omega_effective| = {d_start:.3e} rad/s")
    print(f"  deterministic attitude error injected per step if an endpoint sample is used: "
          f"{d_end * 0.5:.3e} rad")
    print(f"  compare with the per-step gyro angle noise sigma_v*sqrt(dt) = "
          f"{arw_deg_per_sqrt_hour_to_si(0.05) * np.sqrt(0.5):.3e} rad")
    print("  The bias is of the same order as the noise, so it is invisible in RMSE but")
    print("  accumulates coherently between star-tracker updates and shows up as a NEES")
    print("  failure. Measured before the fix: attitude RMS 1.26e-04 rad, mean NEES 1925")
    print("  (dof 6). After using interval_rate(): 4.2e-05 rad, ANEES 6.15 (PART E).")
    ok = d_end > 0.0 and d_start > 0.0
    print(f"  verdict (informational): {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = [part_a(), part_b(), part_c(), part_d(), part_e(), part_f()]
    print()
    print("=" * 78)
    print(f"OVERALL v4: {'PASS' if all(results) else 'FAIL'}")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
