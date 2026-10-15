"""Sensitivity-matrix tests, including the known-answer closed-form check."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.scenarios import (
    LINEAR_THETA_TRUE,
    build_inputs,
    calibration_model,
    design,
    linear_design,
    linear_model,
)
from twincalib.sensitivity import (
    DEFAULT_LOG_STEP,
    fisher_information,
    log_sensitivity,
    predicted_observations,
)


def _thermal_sensitivity(name: str = "baseline-eclipse", **kwargs):
    d = design(name)
    inputs = build_inputs(d)
    return d, log_sensitivity(
        calibration_model(),
        d.theta_true,
        inputs,
        d.initial_state,
        d.channels,
        d.sigma_k,
        d.sample_stride,
        **kwargs,
    )


def test_log_sensitivity_matches_the_closed_form_of_the_linear_model():
    # Known answer: dT/d(ln theta_j) = theta_j dT/dtheta_j, with dT/dtheta from
    # LinearNodeModel.analytic_sensitivity, which is differentiated by hand in
    # that method's docstring.
    model = linear_model()
    inputs = linear_design(121, 25.0)
    sens = log_sensitivity(
        model, LINEAR_THETA_TRUE, inputs, [300.0], [0], 0.3
    )
    exact = model.analytic_sensitivity(
        LINEAR_THETA_TRUE, inputs.times, 300.0
    ) * np.asarray(LINEAR_THETA_TRUE, dtype=float)
    scale = float(np.max(np.abs(exact)))
    # The floor of this comparison is 2.4e-07 relative and is set by the RK4
    # truncation error of simulate against analytic_solution at dt = 25 s, not
    # by the differentiation step; the step study in
    # validation/validate_sensitivity_output.txt shows it flat from 1e-3 down.
    assert np.max(np.abs(sens.jacobian_log - exact)) / scale < 1e-6


def test_weighting_is_exactly_division_by_sigma():
    _d, sens = _thermal_sensitivity()
    assert np.allclose(
        sens.weighted * sens.sigma[:, None], sens.jacobian_log, rtol=0.0, atol=1e-12
    )


def test_fisher_is_symmetric_and_positive_semidefinite():
    _d, sens = _thermal_sensitivity()
    fisher = fisher_information(sens)
    assert np.max(np.abs(fisher - fisher.T)) == pytest.approx(0.0, abs=0.0)
    eig = np.linalg.eigvalsh(fisher)
    assert float(np.min(eig)) > -1e-6 * float(np.max(eig))
    assert np.allclose(fisher, sens.fisher(), rtol=0.0, atol=0.0)


def test_fisher_equals_the_gram_matrix():
    _d, sens = _thermal_sensitivity()
    direct = sens.weighted.T @ sens.weighted
    assert np.max(np.abs(sens.fisher() - direct)) < 1e-9 * float(
        np.max(np.abs(direct))
    )


def test_shapes_and_counts():
    d, sens = _thermal_sensitivity()
    assert sens.n_observations == d.n_observations
    assert sens.n_parameters == 6
    assert sens.jacobian_log.shape == (d.n_observations, 6)
    assert sens.parameter_names == calibration_model().parameter_names
    assert sens.log_step == pytest.approx(DEFAULT_LOG_STEP)


def test_delta_msqr_is_the_column_norm_over_root_n():
    _d, sens = _thermal_sensitivity()
    expected = sens.column_norms() / np.sqrt(sens.n_observations)
    assert np.allclose(sens.delta_msqr(), expected, rtol=1e-12, atol=0.0)


def test_step_size_insensitivity_over_two_decades():
    # The central difference is stable across the flat region of the step
    # study; a 100-fold change in the step must not move the Jacobian by more
    # than a part in 1e5 of its largest entry.
    _d, a = _thermal_sensitivity(log_step=1e-3)
    _d, b = _thermal_sensitivity(log_step=1e-5)
    scale = float(np.max(np.abs(a.jacobian_log)))
    assert np.max(np.abs(a.jacobian_log - b.jacobian_log)) / scale < 1e-5


def test_full_sun_columns_for_the_two_heat_loads_are_proportional():
    # Structural statement, provable by hand: with s(t) = s0 constant the two
    # loads enter only through q_sun s0 + q_par, so
    #   dy/dq_sun = s0 dy/dq_par
    # and in log coordinates q_sun dy/dq_sun = q_sun s0 dy/dq_par, i.e. the two
    # log-columns differ by the constant factor q_sun s0 / q_par.
    d, sens = _thermal_sensitivity("full-sun")
    q_sun, q_par = d.theta_true[4], d.theta_true[5]
    col_sun = sens.jacobian_log[:, 4]
    col_par = sens.jacobian_log[:, 5]
    factor = q_sun * 1.0 / q_par
    scale = float(np.max(np.abs(col_sun)))
    assert np.max(np.abs(col_sun - factor * col_par)) / scale < 1e-7


def test_baseline_columns_for_the_two_heat_loads_are_not_proportional():
    d, sens = _thermal_sensitivity("baseline-eclipse")
    q_sun, q_par = d.theta_true[4], d.theta_true[5]
    col_sun = sens.jacobian_log[:, 4]
    col_par = sens.jacobian_log[:, 5]
    factor = q_sun / q_par
    scale = float(np.max(np.abs(col_sun)))
    assert np.max(np.abs(col_sun - factor * col_par)) / scale > 0.1


def test_predicted_observations_is_the_model_observation():
    d = design("baseline-eclipse")
    inputs = build_inputs(d)
    model = calibration_model()
    a = predicted_observations(
        model, d.theta_true, inputs, d.initial_state, d.channels, d.sample_stride
    )
    b = model.observe(
        d.theta_true, inputs, d.initial_state, d.channels, d.sample_stride
    )
    assert np.max(np.abs(a - b)) == pytest.approx(0.0, abs=0.0)


def test_log_step_validation():
    d = design("baseline-eclipse")
    inputs = build_inputs(d)
    for bad in (0.0, 1.0, -1e-3, np.inf):
        with pytest.raises(ValueError, match="log_step"):
            log_sensitivity(
                calibration_model(),
                d.theta_true,
                inputs,
                d.initial_state,
                d.channels,
                d.sigma_k,
                log_step=bad,
            )


def test_sigma_validation():
    d = design("baseline-eclipse")
    inputs = build_inputs(d)
    with pytest.raises(ValueError, match="sigma"):
        log_sensitivity(
            calibration_model(),
            d.theta_true,
            inputs,
            d.initial_state,
            d.channels,
            0.0,
        )
    with pytest.raises(ValueError, match="sigma"):
        log_sensitivity(
            calibration_model(),
            d.theta_true,
            inputs,
            d.initial_state,
            d.channels,
            np.full(3, 0.3),
        )
