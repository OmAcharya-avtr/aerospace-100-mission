"""Validation 7 — performance benchmark on the 2-core build environment.

Run from the product root::

    PYTHONPATH=src python3 validation/v7_performance.py

All timings are wall-clock, single-threaded (``n_jobs = 1`` throughout; the
gradient-boosting ensemble is single-threaded by construction).  Numbers are
machine-dependent; the *bounds* asserted in ``tests/test_performance.py`` are
set with wide margin so they detect an order-of-magnitude regression rather
than machine-to-machine variation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navbench import (  # noqa: E402
    ExtendedKalmanFilter,
    KalmanFilter,
    LearnedAdaptiveQ,
    MultiplicativeEKF,
    UnscentedKalmanFilter,
    attitude_trajectory,
    circular_orbit_state,
    constant_velocity_2d,
    constant_velocity_cwna,
    generate_adaptive_dataset,
    nees,
    orbit_trajectory,
    quat_from_euler_zyx,
    quat_identity,
    radar_jacobian,
    radar_measurement,
    simulate_linear_system,
    steady_state_riccati,
)


def timed(label: str, fn, unit: str, n_units: int) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    rate = n_units / dt
    print(f"  {label:<46s}{dt:>9.4f} s   {rate:>12.1f} {unit}/s")
    return label, dt, rate


def main() -> int:
    print("=" * 78)
    print("v7 - performance benchmark (single-threaded)")
    print("=" * 78)
    print(f"  numpy {np.__version__}, python {sys.version.split()[0]}")
    print()
    print(f"  {'benchmark':<46s}{'time':>11s}   {'throughput':>12s}")
    print("  " + "-" * 74)

    rng = np.random.default_rng(1)

    # 1. Linear KF.
    f2, q2 = constant_velocity_cwna(1.0, 0.05)
    h2 = np.array([[1.0, 0.0]])
    r2 = np.array([[9.0]])
    _, z2 = simulate_linear_system(f2, h2, q2, r2, np.array([0.0, 1.0]), 20000, rng)
    timed(
        "linear KF, 2 states, 20000 steps",
        lambda: KalmanFilter(f2, h2, q2, r2, np.zeros(2), np.diag([100.0, 10.0])).run(z2),
        "steps", 20000,
    )

    # 2. EKF and UKF on the 4-state radar problem.
    f4, q4 = constant_velocity_2d(1.0, 0.05)
    r4 = np.diag([400.0, 1e-4])
    from navbench import simulate_radar_scenario

    _, z4 = simulate_radar_scenario(
        dt=1.0, n_steps=5000, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
        x0=np.array([8000.0, -5.0, 8000.0, 3.0]), rng=rng,
    )
    p04 = np.diag([300.0, 50.0, 300.0, 50.0])
    x04 = np.array([8000.0, 0.0, 8000.0, 0.0])
    timed(
        "EKF, 4 states, analytic Jacobians, 5000 steps",
        lambda: ExtendedKalmanFilter(
            lambda x: f4 @ x, radar_measurement, q4, r4, x04, p04,
            f_jac=lambda x: f4, h_jac=radar_jacobian,
        ).run(z4),
        "steps", 5000,
    )
    timed(
        "EKF, 4 states, NUMERICAL Jacobians, 2000 steps",
        lambda: ExtendedKalmanFilter(
            lambda x: f4 @ x, radar_measurement, q4, r4, x04, p04
        ).run(z4[:2000]),
        "steps", 2000,
    )
    timed(
        "UKF, 4 states, 9 sigma points, 5000 steps",
        lambda: UnscentedKalmanFilter(
            lambda x: f4 @ x, radar_measurement, q4, r4, x04, p04
        ).run(z4),
        "steps", 5000,
    )

    # 3. MEKF.
    tr = attitude_trajectory(
        inertia=np.diag([10.0, 15.0, 20.0]), quat0=quat_from_euler_zyx(0.2, -0.1, 0.3),
        omega0=np.array([0.01, -0.02, 0.015]), dt=0.5, n_steps=5000,
    )
    rates = tr.interval_rate()
    qm = tr.quat[1:]
    timed(
        "MEKF, 6 error states, 5000 steps",
        lambda: MultiplicativeEKF(
            sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity(),
            p0=np.diag([0.05**2] * 3 + [1e-6**2] * 3),
        ).run(rates, quat_meas=qm, sigma_rad=3e-5, measurement_every=4),
        "steps", 5000,
    )

    # 4. Truth generators.
    timed(
        "rigid-body RK4 attitude, 20000 steps",
        lambda: attitude_trajectory(
            inertia=np.diag([10.0, 15.0, 20.0]), quat0=quat_identity(),
            omega0=np.array([0.01, -0.02, 0.015]), dt=0.1, n_steps=20000,
        ),
        "steps", 20000,
    )
    r0, v0 = circular_orbit_state(500e3)
    timed(
        "two-body RK4 orbit, 20000 steps",
        lambda: orbit_trajectory(position0=r0, velocity0=v0, dt=1.0, n_steps=20000),
        "steps", 20000,
    )

    # 5. Consistency machinery.
    errs = rng.standard_normal((50000, 4))
    covs = np.repeat(np.eye(4)[None, :, :], 50000, axis=0)
    timed("NEES over 50000 samples, 4 states", lambda: nees(errs, covs), "samples", 50000)

    # 6. Riccati.
    timed(
        "steady-state Riccati, 2 states, 200 solves",
        lambda: [steady_state_riccati(f2, h2, q2, r2, tol=1e-13) for _ in range(200)],
        "solves", 200,
    )

    # 7. ML pipeline.
    print()
    print("  ML pipeline (the compute-budget item):")
    t0 = time.perf_counter()
    x, y, _ = generate_adaptive_dataset(n_runs=150, n_steps=400, seed=20260812)
    t_data = time.perf_counter() - t0
    t0 = time.perf_counter()
    LearnedAdaptiveQ(n_members=5, random_state=20260812).fit(x, y)
    t_fit = time.perf_counter() - t0
    print(f"    dataset generation (150 runs, {x.shape[0]} windows): {t_data:.2f} s")
    print(f"    ensemble fit (5 x GradientBoostingRegressor)      : {t_fit:.2f} s")
    print(f"    TOTAL train pipeline                              : {t_data + t_fit:.2f} s")
    print("    budget stated in the build guide                  : 180 s")
    ok = (t_data + t_fit) < 180.0
    print(f"    {'PASS' if ok else 'FAIL'}")

    print()
    print("=" * 78)
    print(f"OVERALL v7: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
