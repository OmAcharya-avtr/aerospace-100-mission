"""Validation 5 — sensor noise models and truth-trajectory conservation laws.

Run from the product root::

    PYTHONPATH=src python3 validation/v5_sensor_and_truth_models.py

PART A  gyro Allan deviation against the analytic IEEE Std 952-2020 form.

The overlapping Allan variance of a rate signal computed from the integrated
angle ``θ`` is (IEEE Std 952-2020, Annex C, Eq. (C.9))

    σ²(τ) = 1 / (2 τ² (N − 2m)) · Σ_{k=1}^{N−2m} (θ_{k+2m} − 2 θ_{k+m} + θ_k)²

with ``τ = m Δt``.  For the model of :class:`navbench.GyroModel` — angle random
walk ``σ_v`` [rad/s^{1/2}] plus rate random walk ``σ_u`` [rad/s^{3/2}] — the
theory gives

    σ_A(τ) = sqrt( σ_v²/τ + σ_u² τ/3 )

i.e. a −1/2 log-log slope at short τ (ARW) and +1/2 at long τ (RRW), with the
minimum at ``τ* = sqrt(3) σ_v/σ_u``.  There is deliberately **no bias-
instability (flicker, slope 0) plateau** in this model; the measured curve
must therefore follow the two-term form everywhere, and the absence of the
flicker floor is a stated limitation of the package, not a defect here.

PART B  unit-vector and position sensor noise statistics against their specs.
PART C  two-body orbit propagator: conservation of specific energy and angular
        momentum, and the Keplerian period ``T = 2π sqrt(a³/μ)``
        (Vallado 2013, Eq. (1-27)).
PART D  airborne coordinated-turn track: the analytic circle radius ``|v|/Ω``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navbench import (  # noqa: E402
    MU_EARTH,
    R_EARTH,
    AccelerometerModel,
    GpsModel,
    GyroModel,
    StarTrackerModel,
    SunSensorModel,
    airborne_trajectory,
    circular_orbit_state,
    dcm_from_quat,
    orbit_trajectory,
    quat_from_euler_zyx,
)


def overlapping_allan(theta: np.ndarray, dt: float, m: int) -> float:
    """Overlapping Allan deviation at cluster size ``m`` from an angle series."""
    n = theta.size
    if n <= 2 * m:
        raise ValueError("series too short for this cluster size")
    d = theta[2 * m :] - 2.0 * theta[m : n - m] + theta[: n - 2 * m]
    tau = m * dt
    return float(np.sqrt(np.sum(d * d) / (2.0 * tau * tau * d.size)))


def part_a() -> bool:
    print("=" * 78)
    print("PART A - gyro Allan deviation vs IEEE Std 952-2020 analytic form")
    print("=" * 78)
    dt = 0.01
    n = 400000
    sigma_v = 1.0e-3  # rad/s^0.5
    sigma_u = 1.0e-3  # rad/s^1.5
    rng = np.random.default_rng(24680)
    gyro = GyroModel(sigma_v=sigma_v, sigma_u=sigma_u, dt=dt, bias0=np.zeros(3))
    rates, _ = gyro.sample_series(np.zeros((n, 3)), rng)
    tau_star = np.sqrt(3.0) * sigma_v / sigma_u
    print(f"  {n} samples at dt = {dt} s ({n * dt:.0f} s), sigma_v = {sigma_v:.3e} rad/s^0.5, "
          f"sigma_u = {sigma_u:.3e} rad/s^1.5")
    print(f"  predicted Allan minimum at tau* = sqrt(3) sigma_v/sigma_u = {tau_star:.4f} s")
    print()
    print(f"  {'tau [s]':>10}{'measured':>14}{'theory':>14}{'ratio':>10}")
    ok = True
    ratios = []
    for m in (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        tau = m * dt
        meas = float(np.mean([overlapping_allan(np.cumsum(rates[:, ax]) * dt, dt, m)
                              for ax in range(3)]))
        theo = float(np.sqrt(sigma_v**2 / tau + sigma_u**2 * tau / 3.0))
        ratio = meas / theo
        ratios.append((tau, ratio))
        print(f"  {tau:>10.2f}{meas:>14.6e}{theo:>14.6e}{ratio:>10.4f}")
    # Tolerance: the Allan estimator's own relative uncertainty is
    # ~1/sqrt(2(N/m - 1)) for overlapping estimates, so long clusters are
    # intrinsically noisy. Judge short/medium clusters tightly and long ones loosely.
    for tau, ratio in ratios:
        rel_unc = 1.0 / np.sqrt(2.0 * (n / (tau / dt) - 1.0))
        tol = max(3.0 * rel_unc, 0.02)
        good = abs(ratio - 1.0) <= tol
        ok &= good
        if not good:
            print(f"    FAIL at tau = {tau:.2f} s: ratio {ratio:.4f}, "
                  f"allowed |ratio-1| <= {tol:.4f}")
    # log-log slopes over the ARW-dominated and RRW-dominated ends
    taus = np.array([2, 5, 10]) * dt
    vals = np.array([np.mean([overlapping_allan(np.cumsum(rates[:, ax]) * dt, dt, m)
                              for ax in range(3)]) for m in (2, 5, 10)])
    s_short = float(np.polyfit(np.log(taus), np.log(vals), 1)[0])
    taus2 = np.array([1000, 2000, 5000]) * dt
    vals2 = np.array([np.mean([overlapping_allan(np.cumsum(rates[:, ax]) * dt, dt, m)
                               for ax in range(3)]) for m in (1000, 2000, 5000)])
    s_long = float(np.polyfit(np.log(taus2), np.log(vals2), 1)[0])
    print()
    print(f"  log-log slope over tau = 0.02-0.10 s : {s_short:+.4f}  (ARW theory -0.5)")
    print(f"  log-log slope over tau = 10-50 s     : {s_long:+.4f}  (RRW theory +0.5)")
    good = abs(s_short + 0.5) < 0.05 and abs(s_long - 0.5) < 0.15
    ok &= good
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def part_b() -> bool:
    print()
    print("=" * 78)
    print("PART B - sensor noise statistics vs specification")
    print("=" * 78)
    ok = True
    rng = np.random.default_rng(13579)
    q_true = quat_from_euler_zyx(0.4, -0.2, 0.9)

    # Star tracker: recover the specified per-axis sigma from the measured
    # small-angle error of each line of sight.
    sigma_st = 5e-5
    refs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1, 1, 1]], dtype=float)
    st = StarTrackerModel(sigma_rad=sigma_st, reference_vectors=refs)
    n = 20000
    errs = []
    rot = dcm_from_quat(q_true).T
    for _ in range(n):
        for obs in st.sample(q_true, rng):
            pred = rot @ obs.reference
            errs.append(np.cross(pred, obs.body))
    errs_arr = np.array(errs)
    # Cross-product error of a unit vector measures the two components of the
    # rotation error perpendicular to the line of sight; the component along
    # the line of sight is unobservable, so the measured variance per axis is
    # 2/3 of the isotropic 3-axis variance when averaged over all directions.
    meas_var = float(np.mean(np.sum(errs_arr**2, axis=1)))
    theo_var = 2.0 * sigma_st**2
    print(f"  star tracker: E|p x b|^2 measured {meas_var:.6e}, theory 2*sigma^2 = "
          f"{theo_var:.6e}, ratio {meas_var / theo_var:.4f}")
    good = abs(meas_var / theo_var - 1.0) < 0.03
    ok &= good
    print(f"    (2 observable components per line of sight)  {'PASS' if good else 'FAIL'}")

    # Star tracker quaternion output.
    from navbench import quat_conjugate, quat_multiply, small_angle_from_quat

    dev = np.array(
        [
            small_angle_from_quat(
                quat_multiply(quat_conjugate(q_true), st.sample_quaternion(q_true, rng))
            )
            for _ in range(20000)
        ]
    )
    meas_sig = np.std(dev, axis=0)
    print(f"  star tracker quaternion output: measured per-axis sigma "
          f"{np.array2string(meas_sig, precision=8)} vs spec {sigma_st:.3e}")
    good = bool(np.all(np.abs(meas_sig / sigma_st - 1.0) < 0.03))
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}")

    # Sun sensor eclipse / FOV behaviour.
    ss = SunSensorModel(sigma_rad=1e-2, sun_vector_inertial=[0.0, 0.0, 1.0],
                        fov_half_angle_rad=np.pi, eclipse_prob=0.3)
    valid = [ss.sample(q_true, rng).valid for _ in range(20000)]
    frac = float(np.mean(valid))
    print(f"  sun sensor eclipse_prob = 0.3: measured valid fraction {frac:.4f} "
          f"(expect 0.700 +/- {3 * np.sqrt(0.3 * 0.7 / 20000):.4f})")
    good = abs(frac - 0.7) < 3 * np.sqrt(0.3 * 0.7 / 20000)
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}")

    # GPS position noise and dropouts.
    gps = GpsModel(sigma_pos=2.5, sigma_vel=0.1, dropout_prob=0.05)
    pos_true = np.array([1000.0, -2000.0, 500.0])
    vel_true = np.array([1.0, 2.0, -0.5])
    samples = [gps.sample(pos_true, rng, vel_true) for _ in range(40000)]
    good_fix = np.array([s.position for s in samples if s.valid])
    good_vel = np.array([s.velocity for s in samples if s.valid])
    drop = 1.0 - len(good_fix) / len(samples)
    sp = np.std(good_fix - pos_true, axis=0)
    sv = np.std(good_vel - vel_true, axis=0)
    print(f"  GPS: dropout measured {drop:.4f} (spec 0.05), position sigma "
          f"{np.array2string(sp, precision=4)} (spec 2.5), velocity sigma "
          f"{np.array2string(sv, precision=5)} (spec 0.1)")
    good = (
        abs(drop - 0.05) < 4 * np.sqrt(0.05 * 0.95 / len(samples))
        and bool(np.all(np.abs(sp / 2.5 - 1.0) < 0.03))
        and bool(np.all(np.abs(sv / 0.1 - 1.0) < 0.03))
    )
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}")

    # Accelerometer: bias plus white noise, in body axes.
    acc = AccelerometerModel(sigma_a=0.02, bias=[0.01, -0.02, 0.03])
    a_i = np.array([0.5, -0.3, 0.2])
    meas = np.array([acc.sample(q_true, a_i, rng) for _ in range(40000)])
    expect_mean = dcm_from_quat(q_true).T @ a_i + np.array([0.01, -0.02, 0.03])
    d_mean = float(np.max(np.abs(np.mean(meas, axis=0) - expect_mean)))
    d_std = float(np.max(np.abs(np.std(meas, axis=0) / 0.02 - 1.0)))
    print(f"  accelerometer: max |mean - (R^T a + b)| = {d_mean:.5f} m/s^2 "
          f"(3*sigma/sqrt(N) = {3 * 0.02 / np.sqrt(40000):.5f}), "
          f"max relative sigma error {d_std:.4f}")
    good = d_mean < 3 * 0.02 / np.sqrt(40000) and d_std < 0.03
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}")
    return ok


def part_c() -> bool:
    print()
    print("=" * 78)
    print("PART C - two-body orbit propagator")
    print("=" * 78)
    ok = True
    for alt, inc in ((500e3, 0.0), (800e3, 0.9), (35786e3, 0.1)):
        r0, v0 = circular_orbit_state(alt, inc)
        a = float(np.linalg.norm(r0))
        period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        n_steps = 1000
        dt = period / n_steps
        tr = orbit_trajectory(position0=r0, velocity0=v0, dt=dt, n_steps=n_steps)
        e = tr.specific_energy()
        h = tr.angular_momentum()
        d_e = float(np.max(np.abs(e - e[0])) / abs(e[0]))
        d_h = float(np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0]))
        close = float(np.linalg.norm(tr.position[-1] - r0)) / a
        e_theory = -MU_EARTH / (2.0 * a)
        d_e_abs = abs(e[0] - e_theory) / abs(e_theory)
        print(f"\n  altitude {alt / 1e3:.0f} km, inclination {inc:.1f} rad, "
              f"a = {a / 1e3:.1f} km, period = {period:.2f} s, dt = {dt:.4f} s")
        print(f"    specific energy: {e[0]:.9e} J/kg vs -mu/(2a) = {e_theory:.9e} "
              f"(relative {d_e_abs:.3e})")
        print(f"    relative energy drift over 1 revolution : {d_e:.3e}  (tol 1e-9)")
        print(f"    relative |h| drift over 1 revolution    : {d_h:.3e}  (tol 1e-9)")
        print(f"    closure |r(T) - r(0)|/a after 1 rev     : {close:.3e}  (tol 1e-7)")
        good = d_e < 1e-9 and d_h < 1e-9 and close < 1e-7 and d_e_abs < 1e-14
        ok &= good
        print(f"    {'PASS' if good else 'FAIL'}")
    print(f"\n  Earth constants used: mu = {MU_EARTH:.9e} m^3/s^2 (IERS 2010 Table 1.1), "
          f"R_E = {R_EARTH:.1f} m (WGS-84)")
    return ok


def part_d() -> bool:
    print()
    print("=" * 78)
    print("PART D - airborne coordinated turn")
    print("=" * 78)
    speed = 200.0
    omega = 0.02
    tr = airborne_trajectory(
        position0=[0.0, 0.0, 3000.0], velocity0=[speed, 0.0, 0.0],
        dt=0.5, n_steps=int(round(2 * np.pi / omega / 0.5)), turn_rate_rad_s=omega,
    )
    radius_theory = speed / omega
    centre = np.array([0.0, radius_theory])
    radii = np.linalg.norm(tr.position[:, :2] - centre, axis=1)
    d_r = float(np.max(np.abs(radii - radius_theory)) / radius_theory)
    d_speed = float(np.max(np.abs(np.linalg.norm(tr.velocity[:, :2], axis=1) - speed)) / speed)
    print(f"  speed {speed} m/s, turn rate {omega} rad/s -> radius |v|/Omega = "
          f"{radius_theory:.4f} m")
    print(f"  max relative radius error over one full circle : {d_r:.3e}  (tol 1e-12)")
    print(f"  max relative ground-speed error                 : {d_speed:.3e}  (tol 1e-12)")
    ok = d_r < 1e-12 and d_speed < 1e-12
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = [part_a(), part_b(), part_c(), part_d()]
    print()
    print("=" * 78)
    print(f"OVERALL v5: {'PASS' if all(results) else 'FAIL'}")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
