"""Unit and known-answer tests for trackbench.control."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trackbench.control import (
    LQRController,
    PIDController,
    bandwidth_estimate,
    benchmark_controllers,
    disturbance_rejection_rms,
    lqr_weights_from_bandwidth,
    pid_gains_from_bandwidth,
    step_response,
    zoh_discretize,
)
from trackbench.dynamics import GimbalAxis, JitterPSD, synthesize_jitter

J, B = 0.05, 0.02
WN = 2.0 * math.pi * 5.0  # 5 Hz design bandwidth [rad/s]


def make_axis(**kw) -> GimbalAxis:
    """Reference axis J = 0.05 kg m^2, b = 0.02 N m s/rad."""
    base = {"inertia": J, "damping": B, "torque_max": 2.0, "rate_max": 1.0}
    base.update(kw)
    return GimbalAxis(**base)


def analytic_second_order(zeta: float, wn: float, t: np.ndarray) -> np.ndarray:
    """Unit step of 1/(s^2/wn^2 + 2 zeta s/wn + 1) (Ogata 2010, eq. 5-12)."""
    wd = wn * math.sqrt(1.0 - zeta**2)
    return 1.0 - np.exp(-zeta * wn * t) * (
        np.cos(wd * t) + zeta / math.sqrt(1.0 - zeta**2) * np.sin(wd * t)
    )


# --------------------------------------------------------------------------
# gain design
# --------------------------------------------------------------------------
def test_pid_gains_known_answer():
    """Hand check: Kp = J wn^2 = 0.05*(10 pi)^2 = 49.3480220; Kd = 2*0.707*J*wn."""
    kp, ki, kd = pid_gains_from_bandwidth(J, WN, 0.707, 0.1)
    assert kp == pytest.approx(0.05 * WN**2, rel=1e-12)
    assert kp == pytest.approx(49.34802200544679, rel=1e-12)
    assert kd == pytest.approx(2 * 0.707 * J * WN, rel=1e-12)
    assert ki == pytest.approx(0.1 * WN * kp, rel=1e-12)


@pytest.mark.parametrize("kw", [
    {"inertia": 0.0}, {"wn": -1.0}, {"zeta": 0.0}, {"integral_alpha": -0.1},
])
def test_pid_gain_validation(kw):
    base = {"inertia": J, "wn": WN, "zeta": 0.707, "integral_alpha": 0.1}
    base.update(kw)
    with pytest.raises(ValueError):
        pid_gains_from_bandwidth(**base)


def test_lqr_weights_known_answer():
    """Hand check: r = q/(J^2 wn^4) = 1/(0.0025 * (10 pi)^4) = 4.10639e-4."""
    q, qr, r = lqr_weights_from_bandwidth(J, WN)
    assert (q, qr) == (1.0, 0.0)
    assert r == pytest.approx(1.0 / (J**2 * WN**4), rel=1e-12)
    assert r == pytest.approx(4.1063929018737e-4, rel=1e-9)


@pytest.mark.parametrize("kw", [{"inertia": -1.0}, {"wn": 0.0}, {"q_angle": 0.0}])
def test_lqr_weight_validation(kw):
    base = {"inertia": J, "wn": WN, "q_angle": 1.0}
    base.update(kw)
    with pytest.raises(ValueError):
        lqr_weights_from_bandwidth(**base)


# --------------------------------------------------------------------------
# PID controller
# --------------------------------------------------------------------------
def test_pid_proportional_only_known_answer():
    """u = Kp e = 2.0 * 1e-3 = 2e-3 N m on the first sample (D term is zero)."""
    pid = PIDController(2.0, 0.0, 0.0, 10.0)
    assert pid.update(1e-3, 0.0, 1e-3) == pytest.approx(2e-3)


def test_pid_integral_accumulates_linearly():
    pid = PIDController(0.0, 1.0, 0.0, 10.0)
    u1 = pid.update(1.0, 0.0, 0.1)
    u2 = pid.update(1.0, 0.0, 0.1)
    assert u1 == pytest.approx(0.1)
    assert u2 == pytest.approx(0.2)


def test_pid_derivative_on_measurement_gives_no_setpoint_kick():
    pid = PIDController(0.0, 0.0, 1.0, 10.0)
    assert pid.update(1.0, 0.0, 1e-3) == pytest.approx(0.0)


def test_pid_derivative_responds_to_measurement_change():
    pid = PIDController(0.0, 0.0, 1.0, 10.0)
    pid.update(0.0, 0.0, 1e-3)
    assert pid.update(0.0, 1e-3, 1e-3) == pytest.approx(-1.0)


def test_pid_output_is_clipped():
    pid = PIDController(1e6, 0.0, 0.0, 2.0)
    assert pid.update(1.0, 0.0, 1e-3) == pytest.approx(2.0)


def test_pid_anti_windup_freezes_integral_in_saturation():
    pid = PIDController(1e6, 1e3, 0.0, 2.0)
    for _ in range(50):
        pid.update(1.0, 0.0, 1e-3)
    frozen = pid.integral
    for _ in range(50):
        pid.update(1.0, 0.0, 1e-3)
    assert pid.integral == pytest.approx(frozen)


def test_pid_integral_unwinds_when_error_reverses():
    # Kp alone cannot saturate the output; the integrator drives it there.
    pid = PIDController(1.0, 1e3, 0.0, 2.0)
    for _ in range(50):
        pid.update(1.0, 0.0, 1e-3)
    saturated = pid.integral
    assert saturated > 0.0
    for _ in range(20):
        pid.update(-1.0, 0.0, 1e-3)
    assert pid.integral < saturated


def test_pid_reset_clears_state():
    pid = PIDController(1.0, 1.0, 1.0, 10.0)
    pid.update(1.0, 0.0, 1e-3)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.update(1.0, 0.0, 1e-3) == pytest.approx(1.0 * 1.0 + 1.0 * 1e-3)


@pytest.mark.parametrize("kw", [
    {"kp": -1.0}, {"ki": -1.0}, {"kd": -1.0}, {"u_max": 0.0}, {"u_max": -1.0},
])
def test_pid_input_validation(kw):
    base = {"kp": 1.0, "ki": 1.0, "kd": 1.0, "u_max": 1.0}
    base.update(kw)
    with pytest.raises(ValueError):
        PIDController(**base)


def test_pid_rejects_bad_dt():
    with pytest.raises(ValueError):
        PIDController(1.0, 0.0, 0.0, 1.0).update(1.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# PD step response vs analytic second-order (hand-checkable case)
# --------------------------------------------------------------------------
def test_pd_step_response_matches_analytic_overshoot_and_peak_time():
    """Closed loop with PD (D on measurement) is exactly

        J s^2 + (Kd + b) s + Kp,  wn = sqrt(Kp/J), zeta = (Kd+b)/(2 sqrt(Kp J))

    so Mp = exp(-pi zeta / sqrt(1-zeta^2)) and tp = pi/(wn sqrt(1-zeta^2)).
    For Kp = 49.3480, Kd = 2.22111, b = 0.02, J = 0.05:
      wn = 31.41593 rad/s, zeta = 0.7133662, Mp = 0.0408453, tp = 0.1426958 s.
    """
    kp, _, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    _, _, m = step_response(make_axis(), PIDController(kp, 0.0, kd, 2.0), 1e-4, 1e-4, 1.0)
    wn = math.sqrt(kp / J)
    zeta = (kd + B) / (2 * math.sqrt(kp * J))
    assert wn == pytest.approx(31.41592653589793, rel=1e-9)
    assert zeta == pytest.approx(0.7133661977236757, rel=1e-9)
    mp = math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2))
    tp = math.pi / (wn * math.sqrt(1 - zeta**2))
    assert m.overshoot == pytest.approx(mp, rel=0.02)
    assert m.peak_time == pytest.approx(tp, rel=0.01)


def test_pd_step_response_matches_analytic_rise_time():
    kp, _, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    _, _, m = step_response(make_axis(), PIDController(kp, 0.0, kd, 2.0), 1e-4, 1e-4, 1.0)
    wn = math.sqrt(kp / J)
    zeta = (kd + B) / (2 * math.sqrt(kp * J))
    t = np.linspace(0.0, 1.0, 200001)
    y = analytic_second_order(zeta, wn, t)
    tr = t[int(np.argmax(y >= 0.9))] - t[int(np.argmax(y >= 0.1))]
    assert m.rise_time == pytest.approx(tr, rel=0.02)


def test_pd_step_trajectory_matches_analytic_pointwise():
    kp, _, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    t, y, _ = step_response(make_axis(), PIDController(kp, 0.0, kd, 2.0), 1e-4, 1e-4, 1.0)
    wn = math.sqrt(kp / J)
    zeta = (kd + B) / (2 * math.sqrt(kp * J))
    y_ref = 1e-4 * analytic_second_order(zeta, wn, t)
    assert np.max(np.abs(y - y_ref)) < 0.01 * 1e-4


def test_higher_damping_reduces_overshoot():
    kp1, _, kd1 = pid_gains_from_bandwidth(J, WN, 0.5)
    kp2, _, kd2 = pid_gains_from_bandwidth(J, WN, 1.0)
    _, _, m1 = step_response(make_axis(), PIDController(kp1, 0.0, kd1, 2.0), 1e-4, 1e-4, 1.0)
    _, _, m2 = step_response(make_axis(), PIDController(kp2, 0.0, kd2, 2.0), 1e-4, 1e-4, 1.0)
    assert m1.overshoot > m2.overshoot
    assert m2.overshoot < 0.01


def test_pid_steady_state_error_is_negligible():
    kp, ki, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    _, _, m = step_response(make_axis(), PIDController(kp, ki, kd, 2.0), 1e-4, 1e-4, 2.0)
    assert abs(m.steady_state_error) < 1e-4 * 1e-3


def test_step_response_rejects_bad_timing():
    with pytest.raises(ValueError):
        step_response(make_axis(), PIDController(1.0, 0, 0, 1.0), 1e-4, 0.0, 1.0)


# --------------------------------------------------------------------------
# LQR
# --------------------------------------------------------------------------
def test_lqr_poles_match_butterworth_design():
    """Weights from eq. (10) place |poles| = wn with zeta = sqrt(2)/2."""
    q, qr, r = lqr_weights_from_bandwidth(J, WN)
    lqr = LQRController(make_axis(damping=0.0), q_angle=q, q_rate=qr, r_torque=r)
    poles = lqr.closed_loop_poles
    assert abs(poles[0]) == pytest.approx(WN, rel=1e-9)
    assert -poles[0].real / abs(poles[0]) == pytest.approx(math.sqrt(0.5), rel=1e-6)


def test_lqr_gain_solves_riccati_equation():
    q, qr, r = lqr_weights_from_bandwidth(J, WN)
    lqr = LQRController(make_axis(), q_angle=q, q_rate=qr, r_torque=r)
    a, b = lqr.axis.state_space()
    p = lqr.riccati_p
    residual = a.T @ p + p @ a - p @ b @ np.linalg.inv([[r]]) @ b.T @ p + np.diag([q, qr])
    assert np.max(np.abs(residual)) < 1e-6 * max(1.0, np.max(np.abs(p)))


def test_lqr_closed_loop_is_stable():
    q, qr, r = lqr_weights_from_bandwidth(J, WN)
    lqr = LQRController(make_axis(), q_angle=q, q_rate=qr, r_torque=r)
    assert np.all(lqr.closed_loop_poles.real < 0)


def test_lqr_step_response_is_well_damped():
    q, qr, r = lqr_weights_from_bandwidth(J, WN)
    lqr = LQRController(make_axis(), q_angle=q, q_rate=qr, r_torque=r)
    _, _, m = step_response(make_axis(), lqr, 1e-4, 1e-4, 1.0)
    assert m.overshoot < 0.10
    assert m.settling_time < 0.5


def test_lqr_discrete_gain_close_to_continuous():
    q, qr, r = lqr_weights_from_bandwidth(J, WN)
    cont = LQRController(make_axis(), q_angle=q, q_rate=qr, r_torque=r)
    disc = LQRController(make_axis(), q_angle=q, q_rate=qr, r_torque=r, discrete_dt=1e-4)
    assert disc.gain == pytest.approx(cont.gain, rel=0.05)


def test_lqr_higher_r_gives_lower_gain():
    lo = LQRController(make_axis(), q_angle=1.0, r_torque=1e-4)
    hi = LQRController(make_axis(), q_angle=1.0, r_torque=1e-2)
    assert hi.gain[0] < lo.gain[0]


@pytest.mark.parametrize("kw", [
    {"q_angle": 0.0}, {"q_rate": -1.0}, {"r_torque": 0.0}, {"discrete_dt": 0.0},
])
def test_lqr_input_validation(kw):
    base = {"q_angle": 1.0, "q_rate": 0.0, "r_torque": 1e-3}
    base.update(kw)
    with pytest.raises(ValueError):
        LQRController(make_axis(), **base)


def test_lqr_rejects_non_axis():
    with pytest.raises(TypeError):
        LQRController("not-an-axis")


def test_lqr_reset_clears_rate_estimate():
    lqr = LQRController(make_axis(), q_angle=1.0, r_torque=1e-3)
    lqr.update(0.0, 1e-4, 1e-3)
    lqr.reset()
    assert lqr.update(0.0, 0.0, 1e-3) == pytest.approx(0.0)


def test_lqr_output_is_clipped_to_torque_max():
    lqr = LQRController(make_axis(torque_max=0.5), q_angle=1.0, r_torque=1e-8)
    assert abs(lqr.update(0.0, 1.0, 1e-3)) <= 0.5


def test_zoh_discretize_matches_analytic_double_integrator():
    """For A = [[0,1],[0,0]], B = [0,1]: Ad = [[1,dt],[0,1]], Bd = [dt^2/2, dt]."""
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    b = np.array([[0.0], [1.0]])
    ad, bd = zoh_discretize(a, b, 0.01)
    assert ad == pytest.approx(np.array([[1.0, 0.01], [0.0, 1.0]]), abs=1e-12)
    assert bd.ravel() == pytest.approx(np.array([0.5e-4, 0.01]), abs=1e-12)


# --------------------------------------------------------------------------
# benchmark harness
# --------------------------------------------------------------------------
def test_disturbance_rejection_reduces_rms_below_open_loop():
    kp, ki, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    d = synthesize_jitter(JitterPSD(1e-12, 3.0, 2.0), 8192, 5000.0, np.random.default_rng(3))
    rms = disturbance_rejection_rms(make_axis(), PIDController(kp, ki, kd, 2.0), d, 2e-4)
    assert rms < np.std(d)


def test_disturbance_rejection_validates_input():
    with pytest.raises(ValueError):
        disturbance_rejection_rms(make_axis(), PIDController(1, 0, 0, 1), np.zeros(1), 1e-3)


def test_bandwidth_estimate_near_design_value():
    kp, _, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    bw = bandwidth_estimate(
        make_axis(), PIDController(kp, 0.0, kd, 2.0), 2e-4,
        freqs=np.logspace(math.log10(1.0), math.log10(20.0), 10),
    )
    assert 3.0 < bw < 8.0


def test_bandwidth_increases_with_gain():
    kp1, _, kd1 = pid_gains_from_bandwidth(J, 2 * math.pi * 2.0, 0.707)
    kp2, _, kd2 = pid_gains_from_bandwidth(J, 2 * math.pi * 8.0, 0.707)
    fs = np.logspace(math.log10(0.5), math.log10(30.0), 12)
    bw1 = bandwidth_estimate(make_axis(), PIDController(kp1, 0, kd1, 2.0), 2e-4, freqs=fs)
    bw2 = bandwidth_estimate(make_axis(), PIDController(kp2, 0, kd2, 2.0), 2e-4, freqs=fs)
    assert bw2 > bw1


def test_benchmark_controllers_returns_expected_rows():
    kp, ki, kd = pid_gains_from_bandwidth(J, WN, 0.707)
    q, qr, r = lqr_weights_from_bandwidth(J, WN)
    d = synthesize_jitter(JitterPSD(1e-12, 3.0, 2.0), 4096, 5000.0, np.random.default_rng(3))
    rows = benchmark_controllers(
        make_axis,
        {
            "PID": lambda ax: PIDController(kp, ki, kd, ax.torque_max),
            "LQR": lambda ax: LQRController(ax, q_angle=q, q_rate=qr, r_torque=r),
        },
        dt=2e-4,
        step_duration=1.0,
        disturbance=d,
    )
    assert [r_["name"] for r_ in rows] == ["PID", "LQR"]
    for row in rows:
        assert row["rejection_factor"] > 1.0
        assert row["bandwidth_hz"] > 0.0
        assert 0.0 <= row["overshoot"] < 1.0
