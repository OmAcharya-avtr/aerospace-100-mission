"""Pinned regression tests.

Every falsifying example Hypothesis has found in this repository is recorded
here as an explicit test with literal inputs and the measured value, because
``.hypothesis/`` is gitignored and a counterexample found by one run is lost on
a fresh clone. These are not duplicates of the property tests: they are the
specific numbers that moved a tolerance, and they fail if the numerics change.
"""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.model import Inputs, LinearNodeModel, rk4_steps
from twincalib.scenarios import build_inputs, calibration_model, design
from twincalib.sensitivity import log_sensitivity


def test_regression_rk4_exponential_error_at_rate_two_amplitude_two():
    # Hypothesis counterexample for test_rk4_matches_the_exponential_closed_form:
    # rate = 2.0, y0 = 2.0 on 81 points over [0, 2], i.e. h = 0.025. The
    # measured worst absolute error is 3.995219477470613e-08, which falsified a
    # bound of 1e-08 * (|y0| + 1) = 3.0e-08.
    times = np.linspace(0.0, 2.0, 81)
    out = rk4_steps(lambda _t, y: -2.0 * y, np.array([2.0]), times)
    exact = 2.0 * np.exp(-2.0 * times)
    worst = float(np.max(np.abs(out[:, 0] - exact)))
    assert worst == pytest.approx(3.995219477470613e-08, rel=1e-6)
    assert worst > 1e-8 * (2.0 + 1.0)


def test_regression_linear_closed_form_error_on_a_coarse_fixed_grid():
    # Hypothesis counterexample for test_linear_closed_form_matches_the_integrator
    # in its original form, which used a fixed 10 s grid for every problem:
    # t_sink = 150 K, T0 = 150 K, C = 100 J/K, k = 2.0 W/K, q = 1.0 W. The time
    # constant is 50 s, so h/tau = 0.2 and the worst discrepancy against the
    # closed form is 2.8984769357975892e-06 K. The property test now sets its
    # grid from the problem's own time constant.
    model = LinearNodeModel(t_sink=150.0)
    times = np.arange(0.0, 1501.0, 10.0)
    inputs = Inputs(times, np.zeros(times.size), np.zeros(times.size))
    numeric = model.simulate([100.0, 2.0, 1.0], inputs, [150.0])[:, 0]
    exact = model.analytic_solution([100.0, 2.0, 1.0], times, 150.0)
    worst = float(np.max(np.abs(numeric - exact)))
    assert worst == pytest.approx(2.8984769357975892e-06, rel=1e-6)


def test_regression_linear_closed_form_error_on_the_time_constant_grid():
    # The second counterexample, after the grid was tied to the time constant:
    # C = 100 J/K, k = 1.0 W/K, q = 1.0 W, 20 steps per tau over 10 tau. The
    # worst discrepancy is 1.9976113208031165e-08 K against a 1.0 K excursion,
    # which falsified a relative bound of 1e-08.
    model = LinearNodeModel(t_sink=150.0)
    tau = 100.0 / 1.0
    times = np.arange(0.0, 10.0 * tau, tau / 20.0)
    inputs = Inputs(times, np.zeros(times.size), np.zeros(times.size))
    numeric = model.simulate([100.0, 1.0, 1.0], inputs, [150.0])[:, 0]
    exact = model.analytic_solution([100.0, 1.0, 1.0], times, 150.0)
    worst = float(np.max(np.abs(numeric - exact)))
    assert worst == pytest.approx(1.9976113208031165e-08, rel=1e-5)


def test_regression_full_sun_null_singular_ratio():
    # Not a Hypothesis counterexample but the number the rank tolerance is set
    # against, pinned so that a change in the differentiation or the integrator
    # cannot move it unnoticed: the smallest singular-value ratio of the
    # full-sun design at its true parameters is 1.0562e-11, four decades below
    # the 1e-08 rank cut, and the smallest ratio of baseline-eclipse is
    # 2.519e-03, five decades above it.
    for name, expected in (("full-sun", 1.0562e-11), ("baseline-eclipse", 2.519e-03)):
        d = design(name)
        sens = log_sensitivity(
            calibration_model(),
            d.theta_true,
            build_inputs(d),
            d.initial_state,
            d.channels,
            d.sigma_k,
            d.sample_stride,
        )
        sv = np.linalg.svd(sens.weighted, compute_uv=False)
        assert float(sv[-1] / sv[0]) == pytest.approx(expected, rel=0.05)
