"""The declared plant, its trim, its Jacobian, its discretisation and its metric."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from hilverdict.plant import (
    ALPHA_VALID_RAD,
    DEFAULT_METRIC,
    SHORT_PERIOD,
    Controller,
    DivergenceMetric,
    LoopModel,
    Plant,
    SamplingLaw,
    zoh_matrices,
)


def test_derivative_at_the_origin_with_no_control_is_zero():
    assert SHORT_PERIOD.derivative(np.zeros(2), 0.0) == pytest.approx(
        np.zeros(2), rel=0.0, abs=0.0
    )


def test_derivative_matches_the_hand_written_equations():
    # Hand calculation at alpha = 0.1 rad, q = 0.2 rad/s, u = 0.05 rad:
    #   alpha_dot = -1.20*0.1 + 0.2 + (-0.18)*0.05 = -0.12 + 0.2 - 0.009 = 0.071
    #   q_dot = -9*0.1 + 42*0.001 + (-1.6)*0.2 + (-22)*0.05
    #         = -0.9 + 0.042 - 0.32 - 1.1 = -2.278
    value = SHORT_PERIOD.derivative(np.array([0.1, 0.2]), 0.05)
    assert float(value[0]) == pytest.approx(0.071, rel=1e-14, abs=1e-16)
    assert float(value[1]) == pytest.approx(-2.278, rel=1e-14, abs=1e-16)


def test_derivative_broadcasts_over_a_batch_axis():
    # The batch path is a separate numpy expression from the scalar path the
    # integrator uses, so the two agree to within rounding and not to the last
    # bit. The tolerance is stated rather than assumed: a few units in the last
    # place on a quantity of order 1.
    batch = np.array([[0.0, 0.0], [0.1, 0.2], [-0.1, 0.05]])
    values = SHORT_PERIOD.derivative(batch, 0.05)
    assert values.shape == (3, 2)
    for index in range(3):
        single = SHORT_PERIOD.derivative(batch[index], 0.05)
        assert values[index] == pytest.approx(single, rel=1e-14, abs=1e-14)


def test_derivative_delegates_to_the_scalar_path_exactly():
    generator = np.random.default_rng(0)
    for _ in range(500):
        alpha = float(generator.uniform(-0.3, 0.3))
        rate = float(generator.uniform(-3.0, 3.0))
        control = float(generator.uniform(-0.7, 0.7))
        array = SHORT_PERIOD.derivative(np.array([alpha, rate]), control)
        scalar = SHORT_PERIOD.derivative_scalar(alpha, rate, control)
        assert float(array[0]) == scalar[0]
        assert float(array[1]) == scalar[1]


def test_batch_and_scalar_paths_agree_to_rounding():
    generator = np.random.default_rng(1)
    states = generator.uniform(-0.3, 0.3, size=(2000, 2))
    states[:, 1] *= 10.0
    batch = SHORT_PERIOD.derivative(states, 0.05)
    worst = 0.0
    for index in range(states.shape[0]):
        scalar = SHORT_PERIOD.derivative_scalar(
            float(states[index, 0]), float(states[index, 1]), 0.05
        )
        worst = max(
            worst,
            abs(float(batch[index, 0]) - scalar[0]),
            abs(float(batch[index, 1]) - scalar[1]),
        )
    assert worst < 1e-13


def test_cubic_term_is_odd_in_incidence():
    positive = SHORT_PERIOD.derivative(np.array([0.2, 0.0]), 0.0)
    negative = SHORT_PERIOD.derivative(np.array([-0.2, 0.0]), 0.0)
    assert float(positive[1]) == pytest.approx(-float(negative[1]), rel=1e-15, abs=0.0)


@pytest.mark.parametrize("degrees", [0.0, 0.5, 2.0, 6.0, 10.0, -3.0, -10.0])
def test_trim_is_an_equilibrium_of_the_cubic_plant(degrees):
    state, control = SHORT_PERIOD.trim(math.radians(degrees))
    residual = SHORT_PERIOD.derivative(state, control)
    assert float(np.max(np.abs(residual))) < 1e-14


def test_trim_at_zero_is_the_origin():
    state, control = SHORT_PERIOD.trim(0.0)
    assert state == pytest.approx(np.zeros(2), rel=0.0, abs=0.0)
    assert control == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_trim_matches_equation_four_by_hand():
    # u_trim = -( a (m_a - m_q z_a) + m_a3 a^3 ) / (m_d - m_q z_d)
    alpha = 0.05
    numerator = alpha * (-9.0 - (-1.6) * (-1.2)) + 42.0 * alpha**3
    denominator = -22.0 - (-1.6) * (-0.18)
    expected = -numerator / denominator
    _state, control = SHORT_PERIOD.trim(alpha)
    assert control == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_trim_raises_when_the_denominator_vanishes():
    degenerate = Plant(
        z_alpha=-1.0, z_delta=-1.0, m_alpha=-1.0, m_alpha3=0.0, m_q=-1.0,
        m_delta=1.0, u_max_rad=1.0,
    )
    with pytest.raises(ValueError, match="no solution"):
        degenerate.trim(0.1)


def test_jacobian_at_zero_has_no_cubic_contribution():
    jacobian = SHORT_PERIOD.jacobian(0.0)
    assert float(jacobian[1, 0]) == pytest.approx(
        SHORT_PERIOD.m_alpha, rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize("degrees", [1.0, 5.0, 10.0])
def test_jacobian_incidence_term_is_three_m_alpha3_alpha_squared(degrees):
    alpha = math.radians(degrees)
    jacobian = SHORT_PERIOD.jacobian(alpha)
    expected = SHORT_PERIOD.m_alpha + 3.0 * SHORT_PERIOD.m_alpha3 * alpha**2
    assert float(jacobian[1, 0]) == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_jacobian_upper_row_is_independent_of_incidence():
    first = SHORT_PERIOD.jacobian(0.0)
    second = SHORT_PERIOD.jacobian(0.2)
    assert first[0] == pytest.approx(second[0], rel=0.0, abs=0.0)


def test_jacobian_agrees_with_central_differences():
    state = np.array([0.08, 0.02])
    step = 1e-6
    analytic = SHORT_PERIOD.jacobian(float(state[0]))
    for column in range(2):
        perturbation = np.zeros(2)
        perturbation[column] = step
        forward = SHORT_PERIOD.derivative(state + perturbation, 0.0)
        backward = SHORT_PERIOD.derivative(state - perturbation, 0.0)
        numeric = (forward - backward) / (2.0 * step)
        assert numeric == pytest.approx(analytic[:, column], rel=1e-6, abs=1e-7)


def test_input_matrix_is_the_declared_column():
    assert SHORT_PERIOD.input_matrix == pytest.approx(
        np.array([[-0.18], [-22.0]]), rel=0.0, abs=0.0
    )


@pytest.mark.parametrize("value", [0.0, 0.1, -0.3])
def test_saturate_passes_an_in_range_command_unchanged(value):
    assert SHORT_PERIOD.saturate(value) == pytest.approx(value, rel=0.0, abs=0.0)


def test_saturate_clips_both_signs():
    assert SHORT_PERIOD.saturate(10.0) == pytest.approx(
        SHORT_PERIOD.u_max_rad, rel=0.0, abs=0.0
    )
    assert SHORT_PERIOD.saturate(-10.0) == pytest.approx(
        -SHORT_PERIOD.u_max_rad, rel=0.0, abs=0.0
    )


def test_controller_command_at_trim_is_the_trim_control():
    controller = Controller(gain=(0.2, 1.44))
    state, control = SHORT_PERIOD.trim(math.radians(4.0))
    assert controller.command(SHORT_PERIOD, state, math.radians(4.0)) == pytest.approx(
        control, rel=1e-14, abs=1e-16
    )


def test_controller_reports_saturation():
    controller = Controller(gain=(0.2, 1.44))
    value, clipped = controller.command_with_saturation(
        SHORT_PERIOD, np.array([0.0, 5.0]), 0.0
    )
    assert clipped is True
    assert value == pytest.approx(SHORT_PERIOD.u_max_rad, rel=0.0, abs=0.0)


def test_controller_reports_no_saturation_inside_the_limit():
    controller = Controller(gain=(0.2, 1.44))
    _value, clipped = controller.command_with_saturation(
        SHORT_PERIOD, np.zeros(2), 0.0
    )
    assert clipped is False


def test_sampling_law_rejects_a_non_positive_period():
    with pytest.raises(ValueError, match="period_s must be positive"):
        SamplingLaw(period_s=0.0, deadline_s=0.0)


def test_sampling_law_rejects_a_deadline_past_the_period():
    with pytest.raises(ValueError, match="deadline_s must lie"):
        SamplingLaw(period_s=0.01, deadline_s=0.02)


def test_sampling_law_rejects_a_zero_deadline():
    with pytest.raises(ValueError, match="deadline_s must lie"):
        SamplingLaw(period_s=0.01, deadline_s=0.0)


def test_sampling_law_rejects_zero_substeps():
    with pytest.raises(ValueError, match="substeps must be at least 1"):
        SamplingLaw(period_s=0.01, deadline_s=0.01, substeps=0)


@pytest.mark.parametrize(
    "alpha,rate", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0)]
)
def test_metric_rejects_non_positive_tolerances(alpha, rate):
    with pytest.raises(ValueError, match="must be positive"):
        DivergenceMetric(alpha_tolerance_rad=alpha, rate_tolerance_rad_s=rate)


def test_metric_scaling_and_inverse_are_inverses():
    product = DEFAULT_METRIC.scaling @ DEFAULT_METRIC.inverse_scaling
    assert product == pytest.approx(np.eye(2), rel=0.0, abs=1e-15)


def test_metric_anisotropy_is_symmetric_under_swapping_the_weights():
    first = DivergenceMetric(alpha_tolerance_rad=0.01, rate_tolerance_rad_s=0.5)
    second = DivergenceMetric(alpha_tolerance_rad=0.5, rate_tolerance_rad_s=0.01)
    assert first.anisotropy == pytest.approx(second.anisotropy, rel=0.0, abs=0.0)


def test_metric_scores_one_at_exactly_the_declared_tolerance():
    alpha_only = np.array([DEFAULT_METRIC.alpha_tolerance_rad, 0.0])
    rate_only = np.array([0.0, DEFAULT_METRIC.rate_tolerance_rad_s])
    assert float(DEFAULT_METRIC.norm(alpha_only)) == pytest.approx(
        1.0, rel=1e-15, abs=0.0
    )
    assert float(DEFAULT_METRIC.norm(rate_only)) == pytest.approx(
        1.0, rel=1e-15, abs=0.0
    )


def test_metric_norm_of_both_tolerances_is_root_two():
    both = np.array(
        [DEFAULT_METRIC.alpha_tolerance_rad, DEFAULT_METRIC.rate_tolerance_rad_s]
    )
    assert float(DEFAULT_METRIC.norm(both)) == pytest.approx(
        math.sqrt(2.0), rel=1e-15, abs=0.0
    )


def test_metric_norm_of_zero_is_zero():
    assert float(DEFAULT_METRIC.norm(np.zeros(2))) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_metric_norm_broadcasts():
    batch = np.zeros((5, 2))
    batch[:, 0] = DEFAULT_METRIC.alpha_tolerance_rad
    values = DEFAULT_METRIC.norm(batch)
    assert values.shape == (5,)
    assert values == pytest.approx(np.ones(5), rel=1e-15, abs=0.0)


def test_zoh_state_matrix_is_the_matrix_exponential():
    jacobian = SHORT_PERIOD.jacobian(0.0)
    discrete, _input = zoh_matrices(jacobian, SHORT_PERIOD.input_matrix, 0.01)
    assert discrete == pytest.approx(expm(jacobian * 0.01), rel=1e-14, abs=1e-16)


def test_zoh_input_matrix_matches_the_inverse_formula():
    jacobian = SHORT_PERIOD.jacobian(0.0)
    discrete, input_discrete = zoh_matrices(
        jacobian, SHORT_PERIOD.input_matrix, 0.01
    )
    expected = np.linalg.solve(
        jacobian, (discrete - np.eye(2)) @ SHORT_PERIOD.input_matrix
    )
    assert input_discrete == pytest.approx(expected, rel=1e-12, abs=1e-18)


def test_zoh_at_zero_duration_is_identity_and_zero():
    discrete, input_discrete = zoh_matrices(
        SHORT_PERIOD.jacobian(0.0), SHORT_PERIOD.input_matrix, 0.0
    )
    assert discrete == pytest.approx(np.eye(2), rel=0.0, abs=1e-16)
    assert input_discrete == pytest.approx(np.zeros((2, 1)), rel=0.0, abs=1e-16)


def test_zoh_composes_over_split_durations():
    jacobian = SHORT_PERIOD.jacobian(0.0)
    column = SHORT_PERIOD.input_matrix
    whole_state, whole_input = zoh_matrices(jacobian, column, 0.01)
    first_state, first_input = zoh_matrices(jacobian, column, 0.004)
    second_state, second_input = zoh_matrices(jacobian, column, 0.006)
    assert second_state @ first_state == pytest.approx(
        whole_state, rel=1e-14, abs=1e-16
    )
    assert second_state @ first_input + second_input == pytest.approx(
        whole_input, rel=1e-13, abs=1e-18
    )


def test_zoh_rejects_a_non_square_jacobian():
    with pytest.raises(ValueError, match="must be square"):
        zoh_matrices(np.ones((2, 3)), np.ones((2, 1)), 0.01)


def test_zoh_rejects_a_mismatched_input_matrix():
    with pytest.raises(ValueError, match="must have shape"):
        zoh_matrices(np.eye(2), np.ones((3, 1)), 0.01)


def test_zoh_rejects_a_negative_duration():
    with pytest.raises(ValueError, match="non-negative"):
        zoh_matrices(np.eye(2), np.ones((2, 1)), -0.01)


def test_loop_model_monodromy_is_a_d_plus_b_d_k(model):
    jacobian = model.plant.jacobian(model.operating_alpha_rad)
    discrete, input_discrete = zoh_matrices(
        jacobian, model.plant.input_matrix, model.sampling.period_s
    )
    expected = discrete + input_discrete @ model.controller.gain_row
    assert model.monodromy() == pytest.approx(expected, rel=1e-14, abs=1e-18)


def test_loop_model_monodromy_is_cached(model):
    first = model.monodromy()
    second = model.monodromy()
    assert first is second


def test_scaled_monodromy_is_a_similarity_transform(model):
    scaled = model.scaled_monodromy()
    recovered = model.metric.inverse_scaling @ scaled @ model.metric.scaling
    assert recovered == pytest.approx(model.monodromy(), rel=1e-12, abs=1e-18)


def test_similarity_preserves_the_spectrum(model):
    first = np.sort(np.abs(np.linalg.eigvals(model.monodromy())))
    second = np.sort(np.abs(np.linalg.eigvals(model.scaled_monodromy())))
    assert first == pytest.approx(second, rel=1e-12, abs=1e-15)


def test_declared_closed_loop_is_stable(model):
    assert model.spectral_radius() < 1.0


def test_scaled_input_matrix_is_s_times_b(model):
    expected = model.metric.scaling @ model.plant.input_matrix
    assert model.scaled_input_matrix() == pytest.approx(expected, rel=0.0, abs=0.0)


def test_declared_validity_limit_is_ten_degrees():
    assert math.degrees(ALPHA_VALID_RAD) == pytest.approx(10.0, rel=1e-15, abs=0.0)


def test_default_metric_is_the_declared_one():
    assert math.degrees(DEFAULT_METRIC.alpha_tolerance_rad) == pytest.approx(
        0.01, rel=1e-14, abs=0.0
    )
    assert math.degrees(DEFAULT_METRIC.rate_tolerance_rad_s) == pytest.approx(
        2.0, rel=1e-14, abs=0.0
    )
    assert DEFAULT_METRIC.anisotropy == pytest.approx(200.0, rel=1e-13, abs=0.0)


def test_loop_model_defaults_are_the_declared_ones():
    loop = LoopModel()
    assert loop.plant is SHORT_PERIOD
    assert loop.controller.gain == (0.20, 1.44)
    assert loop.sampling.period_s == pytest.approx(0.010, rel=0.0, abs=0.0)
    assert loop.sampling.deadline_s == pytest.approx(0.010, rel=0.0, abs=0.0)
    assert loop.sampling.substeps == 4
