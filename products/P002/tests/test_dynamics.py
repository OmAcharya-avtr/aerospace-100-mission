"""Unit tests for trackbench.dynamics (plant, jitter synthesis, sensor)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trackbench.dynamics import (
    AngleSensor,
    GimbalAxis,
    JitterPSD,
    TwoAxisGimbal,
    synthesize_jitter,
    welch_psd,
)


def make_axis(**kw) -> GimbalAxis:
    """Reference axis: J = 0.05 kg m^2, b = 0.02 N m s/rad, 2 N m, 1 rad/s."""
    base = {"inertia": 0.05, "damping": 0.02, "torque_max": 2.0, "rate_max": 1.0}
    base.update(kw)
    return GimbalAxis(**base)


# --------------------------------------------------------------------------
# GimbalAxis
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"inertia": 0.0}, {"inertia": -1.0}, {"damping": -1e-3},
    {"torque_max": 0.0}, {"rate_max": -1.0}, {"accel_max": 0.0},
])
def test_axis_input_validation(kw):
    with pytest.raises(ValueError):
        make_axis(**kw)


def test_state_space_matches_equation_2():
    ax = make_axis()
    a, b = ax.state_space()
    assert a == pytest.approx(np.array([[0.0, 1.0], [0.0, -0.4]]))
    assert b == pytest.approx(np.array([[0.0], [20.0]]))


def test_mechanical_time_constant():
    """Hand check: J/b = 0.05/0.02 = 2.5 s."""
    assert make_axis().mechanical_time_constant == pytest.approx(2.5)


def test_mechanical_time_constant_undamped_is_infinite():
    assert math.isinf(make_axis(damping=0.0).mechanical_time_constant)


def test_undamped_constant_torque_matches_kinematics():
    """b = 0: theta = 0.5 (tau/J) t^2 exactly; RK4 is exact for a quadratic."""
    ax = make_axis(damping=0.0, rate_max=100.0, torque_max=10.0)
    tau, dt, n = 1.0, 1e-3, 500
    for _ in range(n):
        ax.step(tau, dt)
    t = n * dt
    assert ax.angle == pytest.approx(0.5 * (tau / ax.inertia) * t**2, rel=1e-9)
    assert ax.rate == pytest.approx((tau / ax.inertia) * t, rel=1e-9)


def test_damped_step_matches_analytic_first_order_rate():
    """rate(t) = (tau/b)(1 - exp(-b t / J)), exact solution of eq. (1)."""
    ax = make_axis(rate_max=100.0)
    tau, dt, n = 0.01, 1e-3, 2000
    for _ in range(n):
        ax.step(tau, dt)
    t = n * dt
    want = (tau / ax.damping) * (1.0 - math.exp(-ax.damping * t / ax.inertia))
    assert ax.rate == pytest.approx(want, rel=1e-6)


def test_torque_saturation_flag_and_value():
    ax = make_axis()
    ax.step(10.0, 1e-3)
    assert ax.saturated_torque
    # achieved acceleration corresponds to the clipped torque, not the command
    assert ax.rate == pytest.approx(2.0 / 0.05 * 1e-3, rel=1e-3)


def test_no_saturation_flag_when_within_limits():
    ax = make_axis()
    ax.step(0.5, 1e-3)
    assert not ax.saturated_torque
    assert not ax.saturated_rate


def test_rate_limit_clips_rate():
    ax = make_axis(rate_max=0.01)
    for _ in range(1000):
        ax.step(2.0, 1e-3)
    assert ax.rate == pytest.approx(0.01)
    assert ax.saturated_rate


def test_accel_limit_caps_acceleration():
    ax = make_axis(accel_max=1.0, rate_max=100.0)
    ax.step(2.0, 1e-3)  # unlimited would be 40 rad/s^2
    assert ax.rate == pytest.approx(1.0 * 1e-3, rel=1e-6)


def test_axis_reset_clears_state():
    ax = make_axis()
    ax.step(1.0, 1e-2)
    ax.reset(angle=0.3, rate=-0.1)
    assert (ax.angle, ax.rate) == (0.3, -0.1)
    assert not ax.saturated_torque


def test_axis_rejects_bad_dt_and_torque():
    ax = make_axis()
    with pytest.raises(ValueError):
        ax.step(0.1, 0.0)
    with pytest.raises(ValueError):
        ax.step(float("nan"), 1e-3)


# --------------------------------------------------------------------------
# TwoAxisGimbal
# --------------------------------------------------------------------------
def test_two_axis_steps_both_axes_independently():
    g = TwoAxisGimbal(make_axis(), make_axis(inertia=0.1))
    g.step(np.array([1.0, 1.0]), 1e-2)
    assert g.angles[0] > g.angles[1] > 0  # lighter axis moves further


def test_two_axis_from_config():
    cfg = {
        "az": {"inertia": 0.05, "damping": 0.02, "torque_max": 2.0, "rate_max": 1.0},
        "el": {"inertia": 0.06, "damping": 0.03, "torque_max": 2.0, "rate_max": 1.0},
    }
    g = TwoAxisGimbal.from_config(cfg)
    assert g.el.inertia == 0.06


def test_two_axis_from_config_rejects_bad_input():
    with pytest.raises(ValueError):
        TwoAxisGimbal.from_config({"az": {}})
    with pytest.raises(TypeError):
        TwoAxisGimbal("a", "b")


def test_two_axis_reset_and_saturation_flag():
    g = TwoAxisGimbal(make_axis(), make_axis())
    g.reset((0.1, 0.2))
    assert g.angles == pytest.approx(np.array([0.1, 0.2]))
    assert not g.saturated
    g.step(np.array([100.0, 0.0]), 1e-3)
    assert g.saturated


def test_two_axis_rejects_wrong_torque_shape():
    g = TwoAxisGimbal(make_axis(), make_axis())
    with pytest.raises(ValueError, match="shape"):
        g.step(np.array([1.0]), 1e-3)


# --------------------------------------------------------------------------
# JitterPSD
# --------------------------------------------------------------------------
def test_psd_plateau_and_rolloff():
    psd = JitterPSD(s0=1e-12, f_corner=3.0, order=2.0)
    assert psd(np.array([0.0]))[0] == pytest.approx(1e-12)
    # at f = f_corner the second-order model is exactly S0/2
    assert psd(np.array([3.0]))[0] == pytest.approx(0.5e-12, rel=1e-12)
    # decade above the corner: (1 + 100)^-1 ~ 1/101
    assert psd(np.array([30.0]))[0] == pytest.approx(1e-12 / 101.0, rel=1e-9)


def test_psd_rejects_negative_frequency():
    with pytest.raises(ValueError):
        JitterPSD(1e-12, 3.0)(np.array([-1.0]))


@pytest.mark.parametrize("kw", [{"s0": 0.0}, {"f_corner": -1.0}, {"order": 0.0}])
def test_psd_input_validation(kw):
    base = {"s0": 1e-12, "f_corner": 3.0, "order": 2.0}
    base.update(kw)
    with pytest.raises(ValueError):
        JitterPSD(**base)


def test_psd_variance_matches_analytic_arctan_integral():
    """For order = 2, int_0^F S0/(1+(f/fc)^2) df = S0 fc arctan(F/fc)."""
    psd = JitterPSD(1e-12, 3.0, 2.0)
    got = psd.variance(200.0)
    want = 1e-12 * 3.0 * math.atan(200.0 / 3.0)
    assert got == pytest.approx(want, rel=1e-4)


# --------------------------------------------------------------------------
# jitter synthesis
# --------------------------------------------------------------------------
def test_synthesized_jitter_psd_matches_target():
    psd = JitterPSD(1e-12, 3.0, 2.0)
    fs, n = 2000.0, 2**16
    x = synthesize_jitter(psd, n, fs, np.random.default_rng(1))
    f, p = welch_psd(x, fs, nperseg=4096)
    band = (f > 0.5) & (f < 100.0)
    ratio = p[band] / psd(f[band])
    assert np.median(ratio) == pytest.approx(1.0, abs=0.15)


def test_synthesized_jitter_variance_matches_psd_integral():
    psd = JitterPSD(4e-12, 5.0, 2.0)
    fs, n = 2000.0, 2**16
    x = synthesize_jitter(psd, n, fs, np.random.default_rng(4))
    assert np.var(x) == pytest.approx(psd.variance(fs / 2), rel=0.1)


def test_synthesized_jitter_is_zero_mean():
    x = synthesize_jitter(JitterPSD(1e-12, 3.0), 4096, 1000.0, np.random.default_rng(2))
    assert abs(np.mean(x)) < 1e-15


def test_synthesized_jitter_is_reproducible():
    a = synthesize_jitter(JitterPSD(1e-12, 3.0), 1024, 1000.0, np.random.default_rng(9))
    b = synthesize_jitter(JitterPSD(1e-12, 3.0), 1024, 1000.0, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_synthesized_jitter_differs_with_seed():
    a = synthesize_jitter(JitterPSD(1e-12, 3.0), 1024, 1000.0, np.random.default_rng(1))
    b = synthesize_jitter(JitterPSD(1e-12, 3.0), 1024, 1000.0, np.random.default_rng(2))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("kw", [{"n": 4}, {"fs": 0.0}])
def test_synthesize_jitter_validation(kw):
    base = {"psd": JitterPSD(1e-12, 3.0), "n": 1024, "fs": 1000.0}
    base.update(kw)
    with pytest.raises(ValueError):
        synthesize_jitter(**base)


def test_synthesize_jitter_rejects_negative_psd():
    with pytest.raises(ValueError, match="non-negative"):
        synthesize_jitter(lambda f: -np.ones_like(f), 1024, 1000.0)


def test_welch_psd_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        welch_psd(np.zeros((4, 4)), 100.0)


# --------------------------------------------------------------------------
# AngleSensor
# --------------------------------------------------------------------------
def test_sensor_noise_std_matches_nea():
    s = AngleSensor(nea=1e-6)
    rng = np.random.default_rng(0)
    vals = np.array([s.measure(np.zeros(2), rng)[0][0] for _ in range(20000)])
    assert np.std(vals) == pytest.approx(1e-6, rel=0.05)


def test_sensor_zero_nea_is_exact():
    s = AngleSensor(nea=0.0)
    m, ok = s.measure(np.array([1e-4, -2e-4]), np.random.default_rng(0))
    assert ok and m == pytest.approx(np.array([1e-4, -2e-4]))


def test_sensor_quantization_snaps_to_lsb():
    s = AngleSensor(nea=0.0, quantization=1e-6)
    m, _ = s.measure(np.array([1.23e-6, 0.0]), np.random.default_rng(0))
    assert m[0] == pytest.approx(1e-6)


def test_sensor_dropout_holds_last_value_and_flags_invalid():
    s = AngleSensor(nea=0.0, dropout_prob=0.999999)
    rng = np.random.default_rng(0)
    m, ok = s.measure(np.array([5e-4, 0.0]), rng)
    assert not ok
    assert m == pytest.approx(np.zeros(2))


def test_sensor_reset_clears_hold():
    s = AngleSensor(nea=0.0)
    s.measure(np.array([1e-3, 1e-3]), np.random.default_rng(0))
    s.reset()
    s2 = AngleSensor(nea=0.0, dropout_prob=0.999999)
    m, ok = s2.measure(np.array([1e-3, 0.0]), np.random.default_rng(0))
    assert not ok and m == pytest.approx(np.zeros(2))


@pytest.mark.parametrize("kw", [{"nea": -1e-6}, {"quantization": 0.0}, {"dropout_prob": 1.0}])
def test_sensor_input_validation(kw):
    base = {"nea": 1e-6}
    base.update(kw)
    with pytest.raises(ValueError):
        AngleSensor(**base)


def test_sensor_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        AngleSensor(1e-6).measure(np.zeros(3), np.random.default_rng(0))


# --------------------------------------------------------------------------
# property-based
# --------------------------------------------------------------------------
@given(
    tau=st.floats(min_value=-1.0, max_value=1.0),
    n=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=30, deadline=None)
def test_property_rate_never_exceeds_limit(tau, n):
    ax = make_axis(rate_max=0.05)
    for _ in range(n):
        ax.step(tau, 1e-2)
    assert abs(ax.rate) <= 0.05 + 1e-15


@given(j=st.floats(min_value=1e-3, max_value=1.0))
@settings(max_examples=30, deadline=None)
def test_property_state_space_eigenvalues(j):
    ax = make_axis(inertia=j)
    a, _ = ax.state_space()
    ev = np.sort(np.linalg.eigvals(a).real)
    assert ev[0] == pytest.approx(-ax.damping / j)
    assert ev[1] == pytest.approx(0.0, abs=1e-12)
