"""Failure-mode tests: actuator saturation, sensor dropout/quantisation, degenerate cases.

Each test documents the failure being injected and the behaviour the suite
requires: the simulator must degrade in a defined way, never raise an
unhandled exception, and never silently report nominal performance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trackforge.control import (
    LQRController,
    PIDController,
    disturbance_rejection_rms,
    pid_gains_from_bandwidth,
    step_response,
)
from trackforge.dynamics import AngleSensor, GimbalAxis, JitterPSD, synthesize_jitter
from trackforge.reacq import AlwaysLocalPolicy, ReacqConfig, ReacqEnv, evaluate_policy
from trackforge.scan import GaussianUncertainty, simulate_acquisition, spiral_scan
from trackforge.sim import Scenario, run_episode

J, WN = 0.05, 2.0 * math.pi * 5.0


def axis(**kw) -> GimbalAxis:
    """Reference axis, overridable."""
    base = {"inertia": J, "damping": 0.02, "torque_max": 2.0, "rate_max": 1.0}
    base.update(kw)
    return GimbalAxis(**base)


# --------------------------------------------------------------------------
# actuator saturation
# --------------------------------------------------------------------------
def test_torque_starved_axis_cannot_meet_step_but_stays_bounded():
    """Torque limit 1e-4 N m vs a 1e-2 rad step: the loop must not blow up."""
    kp, ki, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    ax = axis(torque_max=1e-4)
    _, y, m = step_response(ax, PIDController(kp, ki, kd, ax.torque_max), 1e-2, 1e-3, 2.0)
    assert np.all(np.isfinite(y))
    assert m.steady_state_error > 0.0  # setpoint not reached
    assert np.max(y) <= 1e-2 * 1.5


def test_anti_windup_limits_overshoot_under_saturation():
    """With anti-windup the saturated response must overshoot far less."""
    kp, ki, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    ki_big = 50.0 * ki

    class _NoAntiWindup(PIDController):
        def update(self, setpoint: float, measurement: float, dt: float) -> float:
            e = setpoint - measurement
            self._integral += self.ki * e * dt
            d = -self.kd * (measurement - (self._prev_y if self._prev_y is not None
                                           else measurement)) / dt
            self._prev_y = measurement
            return float(np.clip(self.kp * e + self._integral + d, -self.u_max, self.u_max))

    _, _, m_aw = step_response(axis(torque_max=0.05),
                               PIDController(kp, ki_big, kd, 0.05), 1e-3, 1e-4, 4.0)
    _, _, m_no = step_response(axis(torque_max=0.05),
                               _NoAntiWindup(kp, ki_big, kd, 0.05), 1e-3, 1e-4, 4.0)
    assert m_aw.overshoot < m_no.overshoot


def test_rate_limited_axis_slews_at_the_limit():
    ax = axis(rate_max=1e-3, torque_max=5.0)
    for _ in range(2000):
        ax.step(5.0, 1e-3)
    assert ax.rate == pytest.approx(1e-3)
    assert ax.saturated_rate


def test_acceleration_limit_degrades_disturbance_rejection():
    kp, ki, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    d = synthesize_jitter(JitterPSD(1e-10, 3.0, 2.0), 8192, 5000.0, np.random.default_rng(2))
    free = disturbance_rejection_rms(axis(), PIDController(kp, ki, kd, 2.0), d, 2e-4)
    limited = disturbance_rejection_rms(
        axis(accel_max=1e-3), PIDController(kp, ki, kd, 2.0), d, 2e-4
    )
    assert limited > free


def test_saturation_fraction_reported_in_episode():
    sc = Scenario(torque_max=1e-4, spike_amplitude=5e-4)
    res = run_episode(sc, seed=1, keep_series=False)
    assert res.saturation_fraction > 0.0
    assert math.isfinite(res.track_rms_rad)


def test_zero_saturation_in_nominal_scenario():
    res = run_episode(Scenario(), seed=1, keep_series=False)
    assert res.saturation_fraction == 0.0


# --------------------------------------------------------------------------
# sensor failures
# --------------------------------------------------------------------------
def test_sensor_dropout_degrades_but_does_not_break_tracking():
    clean = run_episode(Scenario(sensor_dropout=0.0, spike_amplitude=1e-9), seed=8)
    dropped = run_episode(Scenario(sensor_dropout=0.3, spike_amplitude=1e-9), seed=8)
    assert math.isfinite(dropped.track_rms_rad)
    assert dropped.track_rms_rad >= clean.track_rms_rad


def test_total_sensor_dropout_is_rejected_at_construction():
    with pytest.raises(ValueError):
        AngleSensor(nea=1e-6, dropout_prob=1.0)


def test_extreme_sensor_noise_keeps_loop_bounded():
    res = run_episode(Scenario(nea=1e-4, spike_amplitude=1e-9), seed=2)
    assert np.all(np.isfinite(res.los_error))
    assert res.track_rms_rad > 0


def test_coarse_quantisation_raises_tracking_error():
    fine = run_episode(Scenario(quantization=1e-9, spike_amplitude=1e-9), seed=4)
    coarse = run_episode(Scenario(quantization=2e-5, spike_amplitude=1e-9), seed=4)
    assert coarse.track_rms_rad > fine.track_rms_rad


def test_sensor_dropout_holds_last_measurement():
    s = AngleSensor(nea=0.0, dropout_prob=0.5)
    rng = np.random.default_rng(0)
    held = []
    for _ in range(200):
        m, ok = s.measure(np.array([1e-4, 0.0]), rng)
        if not ok:
            held.append(m[0])
    assert held and all(h in (0.0, 1e-4) for h in held)


# --------------------------------------------------------------------------
# acquisition / reacquisition failures
# --------------------------------------------------------------------------
def test_target_outside_scan_never_acquired():
    u = GaussianUncertainty(1e-4)
    p = spiral_scan(u, 1e-5, containment=0.5)
    assert simulate_acquisition(p, np.array([1e-2, 0.0]), p_dwell=1.0) is None


def test_very_low_dwell_detection_probability_can_exhaust_passes():
    u = GaussianUncertainty(1e-4)
    p = spiral_scan(u, 1e-5)
    out = simulate_acquisition(p, np.zeros(2), p_dwell=1e-9,
                               rng=np.random.default_rng(0), max_passes=1)
    assert out is None


def test_reacquisition_timeout_is_reported_not_hidden():
    cfg = ReacqConfig(max_time=1.0, p_detect=1e-6)
    res = evaluate_policy(AlwaysLocalPolicy(), cfg, n_episodes=50, seed=1)
    assert res["success_rate"] == 0.0
    assert res["mean_time_s"] == pytest.approx(cfg.max_time, rel=0.2)


def test_reacquisition_with_zero_drift_still_terminates():
    cfg = ReacqConfig(drift_rate=1e-12, max_time=5.0)
    env = ReacqEnv(cfg)
    env.reset(seed=0)
    steps = 0
    while not env.done and steps < 10000:
        env.step(0)
        steps += 1
    assert env.done


def test_unreachable_cone_still_terminates_by_timeout():
    """Target thrown far outside the cone: FULL scans can never cover it."""
    cfg = ReacqConfig(kappa=500.0, cone_radius=1e-5, max_time=3.0)
    res = evaluate_policy(AlwaysLocalPolicy(), cfg, n_episodes=30, seed=2)
    assert res["mean_time_s"] <= cfg.max_time + 1e-9


# --------------------------------------------------------------------------
# numerical robustness
# --------------------------------------------------------------------------
def test_large_dt_does_not_produce_nan():
    ax = axis()
    for _ in range(100):
        ax.step(1.0, 0.1)
    assert math.isfinite(ax.angle) and math.isfinite(ax.rate)


def test_undamped_lqr_is_still_solvable():
    lqr = LQRController(axis(damping=0.0), q_angle=1.0, r_torque=1e-3)
    assert np.all(lqr.closed_loop_poles.real < 0)


def test_zero_jitter_psd_amplitude_yields_zero_series():
    x = synthesize_jitter(lambda f: np.zeros_like(f), 1024, 1000.0, np.random.default_rng(0))
    assert np.allclose(x, 0.0)
