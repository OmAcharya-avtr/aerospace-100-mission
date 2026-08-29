"""Performance guard rails.

The bounds are deliberately loose — roughly 5-10x the measured value on the
2-core build machine (see ``validation/v7_performance_output.txt``) — so they
catch an order-of-magnitude regression without failing on a slower host.
Measured values on the build machine are quoted in each test.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from navbench import (
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
    quat_identity,
    radar_jacobian,
    radar_measurement,
    simulate_linear_system,
    simulate_radar_scenario,
    steady_state_riccati,
)


def _rate(fn, n_units: float) -> float:
    t0 = time.perf_counter()
    fn()
    return n_units / (time.perf_counter() - t0)


def test_linear_kf_throughput_above_1000_steps_per_second(rng):
    """Measured 8035 steps/s on the build machine."""
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 4000, rng)
    rate = _rate(
        lambda: KalmanFilter(f, h, q, r, np.zeros(2), np.diag([100.0, 10.0])).run(z), 4000
    )
    assert rate > 1000.0


def test_ekf_throughput_above_1000_steps_per_second(rng):
    """Measured 8127 steps/s with analytic Jacobians."""
    f, q = constant_velocity_2d(1.0, 0.05)
    r = np.diag([400.0, 1e-4])
    _, z = simulate_radar_scenario(
        dt=1.0, n_steps=2000, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
        x0=np.array([8000.0, -5.0, 8000.0, 3.0]), rng=rng,
    )
    p0 = np.diag([300.0, 50.0, 300.0, 50.0])
    x0 = np.array([8000.0, 0.0, 8000.0, 0.0])
    rate = _rate(
        lambda: ExtendedKalmanFilter(
            lambda x: f @ x, radar_measurement, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=radar_jacobian,
        ).run(z),
        2000,
    )
    assert rate > 1000.0


def test_ukf_throughput_above_500_steps_per_second(rng):
    """Measured 4003 steps/s (9 sigma points, 4 states)."""
    f, q = constant_velocity_2d(1.0, 0.05)
    r = np.diag([400.0, 1e-4])
    _, z = simulate_radar_scenario(
        dt=1.0, n_steps=2000, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
        x0=np.array([8000.0, -5.0, 8000.0, 3.0]), rng=rng,
    )
    p0 = np.diag([300.0, 50.0, 300.0, 50.0])
    x0 = np.array([8000.0, 0.0, 8000.0, 0.0])
    rate = _rate(
        lambda: UnscentedKalmanFilter(
            lambda x: f @ x, radar_measurement, q, r, x0, p0
        ).run(z),
        2000,
    )
    assert rate > 500.0


def test_mekf_throughput_above_800_steps_per_second():
    """Measured 6814 steps/s (6 error states, update every 4th step)."""
    tr = attitude_trajectory(
        inertia=np.diag([10.0, 15.0, 20.0]), quat0=quat_identity(),
        omega0=np.array([0.01, -0.02, 0.015]), dt=0.5, n_steps=2000,
    )
    rates = tr.interval_rate()
    qm = tr.quat[1:]
    rate = _rate(
        lambda: MultiplicativeEKF(
            sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity(),
            p0=np.diag([0.05**2] * 3 + [1e-6**2] * 3),
        ).run(rates, quat_meas=qm, sigma_rad=3e-5, measurement_every=4),
        2000,
    )
    assert rate > 800.0


def test_attitude_truth_throughput_above_400_steps_per_second():
    """Measured 3459 steps/s (RK4, 4 derivative evaluations per step)."""
    rate = _rate(
        lambda: attitude_trajectory(
            inertia=np.diag([10.0, 15.0, 20.0]), quat0=quat_identity(),
            omega0=np.array([0.01, -0.02, 0.015]), dt=0.1, n_steps=4000,
        ),
        4000,
    )
    assert rate > 400.0


def test_orbit_truth_throughput_above_8000_steps_per_second():
    """Measured 61303 steps/s."""
    r0, v0 = circular_orbit_state(500e3)
    rate = _rate(
        lambda: orbit_trajectory(position0=r0, velocity0=v0, dt=1.0, n_steps=8000), 8000
    )
    assert rate > 8000.0


def test_nees_throughput_above_15000_samples_per_second(rng):
    """Measured 105670 samples/s for a 4-state error."""
    errs = rng.standard_normal((20000, 4))
    covs = np.repeat(np.eye(4)[None, :, :], 20000, axis=0)
    assert _rate(lambda: nees(errs, covs), 20000) > 15000.0


def test_riccati_solve_under_100_ms():
    """Measured 5.7 ms per solve (176.6 solves/s) at tol = 1e-13."""
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    t0 = time.perf_counter()
    steady_state_riccati(f, h, q, r, tol=1e-13)
    assert time.perf_counter() - t0 < 0.1


@pytest.mark.parametrize("n_runs", [30])
def test_ml_pipeline_within_compute_budget(n_runs):
    """Dataset + fit for 150 runs measured at 12.0 s total; the build guide's
    budget is 180 s. This scaled-down version must stay well under 30 s."""
    t0 = time.perf_counter()
    x, y, _ = generate_adaptive_dataset(n_runs=n_runs, n_steps=400, seed=20260812)
    LearnedAdaptiveQ(n_members=3, n_estimators=60, random_state=1).fit(x, y)
    assert time.perf_counter() - t0 < 30.0
