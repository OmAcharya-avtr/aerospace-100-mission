"""Simulator known answers, validation, determinism, and the analytic reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from falsifyloop.requirements import Signal
from falsifyloop.systems import (
    DEFAULT_DT,
    DEFAULT_HORIZON,
    LoopInput,
    LoopParameters,
    simulate,
    simulate_linear_zoh,
)

NOMINAL = LoopInput(
    step_amplitude=5.0,
    kp_factor=1.0,
    kd_factor=1.0,
    tau_factor=1.0,
    gust_amplitude=0.0,
    gust_frequency=1.0,
)


def test_nominal_modes_known_answer() -> None:
    # wn = sqrt(M_delta * Kp) = sqrt(18 * 1.4) = sqrt(25.2) = 5.019960...
    # zeta = (M_delta * Kd - M_q) / (2 wn) = (18*0.35 + 1.2) / (2*5.01996)
    #      = 7.5 / 10.03992 = 0.747018...
    wn, zeta = LoopParameters().nominal_modes()
    assert wn == pytest.approx(math.sqrt(25.2), rel=1e-12)
    assert zeta == pytest.approx(7.5 / (2.0 * math.sqrt(25.2)), rel=1e-12)


def test_trace_shape_and_signal_names() -> None:
    trace = simulate(NOMINAL)
    # horizon 2.0 s at dt 0.005 s gives floor(2.0/0.005) + 1 = 401 samples.
    assert trace.length == 401
    assert trace.dt == pytest.approx(DEFAULT_DT)
    assert set(trace.names) == {"cmd", "delta", "error", "over", "q", "theta"}


def test_initial_condition_known_answer() -> None:
    # The loop starts from rest, so theta[0] = q[0] = delta[0] = 0, and the
    # first command is Kp * step = 1.4 * 5 = 7.0 deg, inside the 20 deg limit.
    trace = simulate(NOMINAL)
    assert trace.signal("theta")[0] == 0.0
    assert trace.signal("q")[0] == 0.0
    assert trace.signal("delta")[0] == 0.0
    assert trace.signal("cmd")[0] == pytest.approx(7.0)
    assert trace.signal("error")[0] == pytest.approx(5.0)
    assert trace.signal("over")[0] == pytest.approx(-5.0)


def test_derived_signals_are_consistent_with_theta() -> None:
    trace = simulate(NOMINAL)
    theta = trace.signal("theta")
    np.testing.assert_allclose(trace.signal("error"), NOMINAL.step_amplitude - theta)
    np.testing.assert_allclose(trace.signal("over"), theta - NOMINAL.step_amplitude)


def test_nominal_loop_settles_near_the_commanded_step() -> None:
    # zeta = 0.747, wn = 5.02 rad/s, so the 2 % settling time is about
    # 4/(zeta*wn) = 1.07 s, comfortably inside the 2 s horizon.
    trace = simulate(NOMINAL)
    assert abs(trace.signal("theta")[-1] - 5.0) < 0.05


def test_deflection_and_rate_limits_are_respected() -> None:
    params = LoopParameters()
    aggressive = LoopInput(6.0, 2.0, 0.25, 0.6, 15.0, 1.0)
    trace = simulate(aggressive)
    delta = trace.signal("delta")
    assert np.all(np.abs(delta) <= params.deflection_limit + 1e-12)
    rate = np.abs(np.diff(delta)) / trace.dt
    assert np.all(rate <= params.rate_limit + 1e-9)
    assert np.all(np.abs(trace.signal("cmd")) <= params.deflection_limit + 1e-12)


def test_simulation_is_deterministic() -> None:
    first = simulate(NOMINAL).signal("theta")
    second = simulate(NOMINAL).signal("theta")
    np.testing.assert_array_equal(first, second)


def test_zero_gust_and_zero_step_stays_at_rest() -> None:
    at_rest = LoopInput(1e-300, 1.0, 1.0, 1.0, 0.0, 1.0)
    trace = simulate(at_rest)
    assert np.max(np.abs(trace.signal("theta"))) < 1e-290


def test_euler_matches_the_exact_zoh_solution_in_the_unsaturated_regime() -> None:
    # A small step with zero gust never hits either clip, so the exact
    # matrix-exponential solution applies and the Euler scheme must track it to
    # the first-order global error the module docstring claims.
    gentle = LoopInput(1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    euler = simulate(gentle, dt=0.001, horizon=1.0)
    exact = simulate_linear_zoh(gentle, dt=0.001, horizon=1.0)
    assert np.all(np.abs(euler.signal("delta")) < LoopParameters().deflection_limit)
    worst = np.max(np.abs(euler.signal("theta") - exact.signal("theta")))
    assert worst < 5e-3, f"worst theta discrepancy {worst:.6g} deg"


def test_euler_self_converges_as_dt_shrinks() -> None:
    gentle = LoopInput(1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    coarse = simulate(gentle, dt=0.004, horizon=1.0)
    fine = simulate(gentle, dt=0.0005, horizon=1.0)
    # Compare at the coarse grid's sample times, which are a subset of the fine
    # grid's because 0.004 / 0.0005 = 8 exactly.
    sub = fine.signal("theta")[::8]
    worst = np.max(np.abs(coarse.signal("theta") - sub[: coarse.length]))
    assert worst < 2e-2, f"worst self-convergence discrepancy {worst:.6g} deg"


def test_loop_input_round_trips_through_an_array() -> None:
    vector = NOMINAL.to_array()
    assert vector.shape == (6,)
    assert LoopInput.from_array(vector) == NOMINAL
    assert LoopInput.FIELDS[0] == "step_amplitude"


def test_signal_term_reads_the_simulated_trace() -> None:
    trace = simulate(NOMINAL)
    np.testing.assert_allclose(Signal("theta").values(trace), trace.signal("theta"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kp_factor": 0.0}, "kp_factor"),
        ({"kd_factor": -1.0}, "kd_factor"),
        ({"tau_factor": 0.0}, "tau_factor"),
        ({"gust_amplitude": -1.0}, "gust_amplitude"),
        ({"gust_frequency": -1.0}, "gust_frequency"),
        ({"step_amplitude": float("nan")}, "finite"),
    ],
)
def test_loop_input_validation(kwargs, match) -> None:
    base = {
        "step_amplitude": 1.0,
        "kp_factor": 1.0,
        "kd_factor": 1.0,
        "tau_factor": 1.0,
        "gust_amplitude": 0.0,
        "gust_frequency": 1.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LoopInput(**base)


def test_loop_input_from_array_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="6 decision variables"):
        LoopInput.from_array(np.zeros(5))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"control_effectiveness": 0.0}, "control_effectiveness"),
        ({"actuator_tau": -0.1}, "actuator_tau"),
        ({"rate_limit": 0.0}, "rate_limit"),
        ({"deflection_limit": -1.0}, "deflection_limit"),
        ({"kp": 0.0}, "kp"),
        ({"pitch_damping": 1.0}, "non-positive"),
        ({"pitch_damping": float("nan")}, "finite"),
        ({"kd": float("inf")}, "finite"),
    ],
)
def test_loop_parameters_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        LoopParameters(**kwargs)


def test_simulate_rejects_bad_timing() -> None:
    with pytest.raises(ValueError, match="dt must be finite"):
        simulate(NOMINAL, dt=0.0)
    with pytest.raises(ValueError, match="horizon must be at least"):
        simulate(NOMINAL, dt=0.01, horizon=0.01)
    # tau_factor 0.6 gives tau = 0.03 s, so dt = 0.05 s exceeds it.
    slow = LoopInput(1.0, 1.0, 1.0, 0.6, 0.0, 1.0)
    with pytest.raises(ValueError, match="actuator time constant"):
        simulate(slow, dt=0.05)


def test_simulate_type_validation() -> None:
    with pytest.raises(TypeError, match="LoopInput"):
        simulate([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LoopParameters"):
        simulate(NOMINAL, parameters={"kp": 1.0})  # type: ignore[arg-type]


def test_default_horizon_and_dt_are_the_documented_values() -> None:
    assert DEFAULT_DT == pytest.approx(0.005)
    assert DEFAULT_HORIZON == pytest.approx(2.0)


def test_gust_drives_the_loop_away_from_the_no_gust_response() -> None:
    quiet = simulate(LoopInput(3.0, 1.0, 1.0, 1.0, 0.0, 1.0))
    gusty = simulate(LoopInput(3.0, 1.0, 1.0, 1.0, 15.0, 1.0))
    assert np.max(np.abs(quiet.signal("theta") - gusty.signal("theta"))) > 0.1
