"""Model tests: the integrator, the closed forms and the input validation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as sc

from twincalib.constants import STEFAN_BOLTZMANN, T_DEEP_SPACE, ZERO_CELSIUS
from twincalib.model import Inputs, LinearNodeModel, ThermalTwinModel, rk4_steps
from twincalib.scenarios import LINEAR_THETA_TRUE, THETA_TRUE, build_inputs, design


def test_stefan_boltzmann_matches_scipy_to_relative_tolerance():
    # The two literals are truncations of the same irrational SI value at
    # different places, so they agree to within rounding and not bit-exactly.
    assert STEFAN_BOLTZMANN == pytest.approx(sc.Stefan_Boltzmann, rel=1e-12)


def test_deep_space_and_ice_point_literals():
    # Fixsen 2009: T_CMB = 2.72548 K. SI Brochure: 0 degC = 273.15 K exactly.
    assert T_DEEP_SPACE == pytest.approx(2.72548, abs=1e-9)
    assert ZERO_CELSIUS == pytest.approx(273.15, abs=1e-12)


def test_inputs_rejects_bad_grids():
    t = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="at least 2"):
        Inputs(np.array([0.0]), np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        Inputs(np.array([0.0, 2.0, 1.0]), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="uniformly spaced"):
        Inputs(np.array([0.0, 1.0, 3.0]), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="same shape"):
        Inputs(t, np.zeros(4), np.zeros(5))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Inputs(t, np.zeros(5), np.full(5, 1.5))


def test_inputs_dt_and_steps():
    inp = Inputs(np.arange(7) * 2.5, np.zeros(7), np.zeros(7))
    assert inp.dt == pytest.approx(2.5)
    assert inp.n_steps == 6


def test_rk4_reproduces_exponential_decay():
    # dy/dt = -y, y(0) = 1 -> y(t) = exp(-t). Hand answer at t = 1: 0.36787944.
    times = np.linspace(0.0, 1.0, 101)
    out = rk4_steps(lambda _t, y: -y, np.array([1.0]), times)
    assert out[-1, 0] == pytest.approx(np.exp(-1.0), rel=1e-10)
    assert out[0, 0] == pytest.approx(1.0, abs=0.0)


def test_rk4_order_is_four_on_a_known_problem():
    # Halving the step must cut the error by about 16 for a fourth-order scheme.
    errors = []
    for n in (20, 40, 80):
        times = np.linspace(0.0, 2.0, n + 1)
        out = rk4_steps(lambda t, y: np.array([np.cos(t)]), np.array([0.0]), times)
        errors.append(abs(float(out[-1, 0]) - np.sin(2.0)))
    for a, b in zip(errors[:-1], errors[1:], strict=True):
        assert 8.0 < a / b < 40.0


def test_linear_node_integrator_matches_closed_form():
    model = LinearNodeModel(t_sink=280.0)
    times = np.arange(0, 3001, 30, dtype=float)
    inp = Inputs(times, np.zeros(times.size), np.zeros(times.size))
    numeric = model.simulate(LINEAR_THETA_TRUE, inp, [300.0])[:, 0]
    exact = model.analytic_solution(LINEAR_THETA_TRUE, times, 300.0)
    # 1.02e-06 K at dt = 30 s against a 320 s time constant; the measured
    # fourth-order convergence of this comparison is in
    # validation/validate_model_output.txt.
    assert np.max(np.abs(numeric - exact)) < 5e-6


def test_linear_node_closed_form_hand_value():
    # C = 800 J/K, k = 2.5 W/K, q = 40 W, T_sink = 280 K, T0 = 300 K.
    # Steady state = 280 + 40/2.5 = 296 K. tau = C/k = 320 s.
    # At t = 320 s: T = 296 + (300 - 296) exp(-1) = 296 + 4*0.3678794 = 297.4715 K.
    model = LinearNodeModel(t_sink=280.0)
    value = model.analytic_solution(LINEAR_THETA_TRUE, np.array([0.0, 320.0]), 300.0)[1]
    assert value == pytest.approx(297.47151776, abs=1e-7)


def test_linear_node_analytic_sensitivity_matches_finite_difference():
    model = LinearNodeModel(t_sink=280.0)
    times = np.arange(0, 2001, 50, dtype=float)
    analytic = model.analytic_sensitivity(LINEAR_THETA_TRUE, times, 300.0)
    theta = np.asarray(LINEAR_THETA_TRUE, dtype=float)
    for j in range(3):
        h = 1e-6 * theta[j]
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        fd = (
            model.analytic_solution(up, times, 300.0)
            - model.analytic_solution(dn, times, 300.0)
        ) / (2.0 * h)
        scale = max(float(np.max(np.abs(analytic[:, j]))), 1e-30)
        assert np.max(np.abs(fd - analytic[:, j])) / scale < 1e-6


def test_thermal_twin_steady_state_hand_value():
    # h_rad (T2^4 - T_space^4) = P + q_sun s + q_par with P = 25 W, s = 0,
    # q_par = 18 W, h_rad = 1.205e-8 W/K^4:
    #   T2 = ((25 + 18)/1.205e-8 + 2.72548^4)^(1/4) = 244.410778 K
    #   T1 = T2 + P/k12 = T2 + 25/2.5 = T2 + 10 K
    model = ThermalTwinModel()
    t1, t2 = model.steady_state(THETA_TRUE, dissipation=25.0, sun=0.0)
    expected_t2 = ((25.0 + 18.0) / 1.205e-8 + T_DEEP_SPACE**4) ** 0.25
    assert t2 == pytest.approx(expected_t2, rel=1e-12)
    assert t1 - t2 == pytest.approx(10.0, rel=1e-12)
    assert t2 == pytest.approx(244.410778, abs=1e-5)


def test_thermal_twin_relaxes_to_the_algebraic_steady_state():
    model = ThermalTwinModel()
    times = np.arange(0, 240001, 60, dtype=float)
    inp = Inputs(times, np.full(times.size, 25.0), np.zeros(times.size))
    final = model.simulate(THETA_TRUE, inp, [250.0, 240.0])[-1]
    t1, t2 = model.steady_state(THETA_TRUE, 25.0, 0.0)
    assert final[0] == pytest.approx(t1, abs=1e-3)
    assert final[1] == pytest.approx(t2, abs=1e-3)


def test_steady_state_rejects_impossible_inputs():
    model = ThermalTwinModel()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.steady_state(THETA_TRUE, 10.0, 1.4)
    with pytest.raises(ValueError, match="must be positive"):
        model.steady_state((800.0, 4000.0, 2.5, 1.205e-8, 1e-12, 1e-12), -5.0, 0.0)


def test_time_constants_bracket_the_physics():
    model = ThermalTwinModel()
    fast, slow = model.time_constants(THETA_TRUE, 280.0)
    assert 0.0 < fast < slow
    # The fast mode is the series capacitance over the interface conductance:
    # (C1 C2/(C1+C2))/k12 = (800*4000/4800)/2.5 = 266.67 s, which the coupled
    # eigenvalue shifts slightly downwards.
    assert 200.0 < fast < 300.0
    assert 3000.0 < slow < 8000.0


def test_time_constants_reject_bad_temperature():
    with pytest.raises(ValueError, match="positive, finite"):
        ThermalTwinModel().time_constants(THETA_TRUE, -1.0)


def test_parameter_validation():
    model = ThermalTwinModel()
    with pytest.raises(ValueError, match="expected 6 parameters"):
        model.check_parameters([1.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        model.check_parameters([800.0, 4000.0, 2.5, 1.205e-8, 55.0, -1.0])
    with pytest.raises(ValueError, match="finite"):
        model.check_parameters([800.0, 4000.0, 2.5, 1.205e-8, 55.0, np.nan])


def test_initial_state_validation():
    model = ThermalTwinModel()
    inp = build_inputs(design("baseline-eclipse"))
    with pytest.raises(ValueError, match="exactly 2 entries"):
        model.simulate(THETA_TRUE, inp, [285.0])
    with pytest.raises(ValueError, match="exactly 1 entry"):
        LinearNodeModel().simulate(LINEAR_THETA_TRUE, inp, [285.0, 275.0])


def test_observe_validation_and_shape():
    model = ThermalTwinModel()
    d = design("baseline-eclipse")
    inp = build_inputs(d)
    out = model.observe(THETA_TRUE, inp, d.initial_state, (0, 1), 1)
    assert out.shape == (2 * d.n_samples,)
    with pytest.raises(ValueError, match="sample_stride"):
        model.observe(THETA_TRUE, inp, d.initial_state, (0,), 0)
    with pytest.raises(ValueError, match="at least one channel"):
        model.observe(THETA_TRUE, inp, d.initial_state, (), 1)
    with pytest.raises(ValueError, match="channels must index"):
        model.observe(THETA_TRUE, inp, d.initial_state, (5,), 1)


def test_observe_stride_subsets_the_grid():
    model = ThermalTwinModel()
    d = design("baseline-eclipse")
    inp = build_inputs(d)
    full = model.observe(THETA_TRUE, inp, d.initial_state, (0,), 1)
    strided = model.observe(THETA_TRUE, inp, d.initial_state, (0,), 4)
    # No interpolation is done: the strided observations are a subset.
    assert np.max(np.abs(strided - full[::4])) == pytest.approx(0.0, abs=1e-12)


def test_unstable_step_is_reported_not_silently_wrong():
    model = ThermalTwinModel()
    times = np.arange(0, 6001, 30, dtype=float)
    inp = Inputs(times, np.full(times.size, 25.0), np.ones(times.size))
    with pytest.raises(ValueError, match="integration diverged"):
        model.simulate((800.0, 4000.0, 400.0, 1.205e-8, 55.0, 18.0), inp, [285.0, 275.0])


def test_discrepancy_flag_and_effect():
    clean = ThermalTwinModel()
    truth = ThermalTwinModel(conductance_slope=1.5)
    assert not clean.has_discrepancy
    assert truth.has_discrepancy
    d = design("baseline-eclipse")
    inp = build_inputs(d)
    a = clean.observe(THETA_TRUE, inp, d.initial_state, d.channels, 1)
    b = truth.observe(THETA_TRUE, inp, d.initial_state, d.channels, 1)
    assert np.max(np.abs(a - b)) > 0.5


def test_model_constructor_validation():
    with pytest.raises(ValueError, match="t_space"):
        ThermalTwinModel(t_space=-1.0)
    with pytest.raises(ValueError, match="conductance_slope"):
        ThermalTwinModel(conductance_slope=np.inf)
    with pytest.raises(ValueError, match="t_reference"):
        ThermalTwinModel(t_reference=0.0)
    with pytest.raises(ValueError, match="t_sink"):
        LinearNodeModel(t_sink=0.0)


def test_twin_model_base_simulate_is_abstract():
    class Bare(type(ThermalTwinModel())().__class__.__bases__[0]):  # pragma: no cover
        pass

    base = ThermalTwinModel.__mro__[1]
    with pytest.raises(NotImplementedError):
        base.simulate(base(), [1.0], None, [1.0])
