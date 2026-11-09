"""The declared plant: known answers, the exact closed form, and the two identities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import (
    ORDER_DURATION,
    ORDER_INITIAL_RATE,
    ORDER_PARAMETERS,
    order_campaign,
)
from twindiverge.plant import (
    CAMPAIGNS,
    NOMINAL,
    OMEGA_MAX,
    OMEGA_MIN,
    SCALE_DIRECTION,
    Campaign,
    WheelParameters,
    perturb,
    simulate,
    simulate_with_sensitivities,
)
from twindiverge.reference import exact_campaign, exact_rate


def test_scale_direction_is_the_unit_diagonal():
    """1/sqrt(5) = 0.4472135954999579, hand-calculable."""
    assert SCALE_DIRECTION == pytest.approx(
        np.full(5, 0.4472135954999579), rel=1e-15, abs=0.0
    )
    assert float(np.linalg.norm(SCALE_DIRECTION)) == pytest.approx(1.0, rel=1e-15, abs=0.0)


def test_derivative_at_the_declared_initial_point_is_the_hand_calculation():
    """tau = 0.032*1 - 0.0025 - 1.2e-5*100 - 2e-8*100^2 = 0.0281 N m.

    omega-dot = 0.0281 / 0.0130 = 2.1615384615384617 rad/s^2. Every term is
    written out so the arithmetic can be checked by hand:
    0.032 - 0.0025 = 0.0295; 0.0295 - 0.0012 = 0.0283; 0.0283 - 0.0002 = 0.0281.
    """
    trajectory = simulate_with_sensitivities(NOMINAL, CAMPAIGNS["full"])
    assert float(trajectory.torque_nm[0]) == pytest.approx(0.0281, rel=1e-14, abs=0.0)
    assert float(trajectory.rate_rad_s[0]) == pytest.approx(100.0, rel=0.0, abs=0.0)
    derivative = float(trajectory.torque_nm[0]) / NOMINAL.inertia_kg_m2
    assert derivative == pytest.approx(2.1615384615384617, rel=1e-13, abs=0.0)


def test_explicit_sensitivities_at_the_initial_point_sum_to_zero_by_hand():
    """Identity A at t = 0, with every term written out.

    g = (-2.1615384615384617, 2.4615384615384617, -0.1923076923076923,
         -0.09230769230769231, -0.015384615384615385) rad/s^2 per unit relative
    change, and 2.4615384615384617 - 0.1923076923076923 - 0.09230769230769231
    - 0.015384615384615385 = 2.1615384615384617 exactly in exact arithmetic.
    """
    inertia = NOMINAL.inertia_kg_m2
    expected = np.array(
        [
            -0.0281 / inertia,
            0.032 / inertia,
            -0.0025 / inertia,
            -1.2e-3 / inertia,
            -2.0e-4 / inertia,
        ]
    )
    assert float(expected[1:].sum()) == pytest.approx(
        -float(expected[0]), rel=1e-13, abs=0.0
    )
    assert float(expected.sum()) == pytest.approx(0.0, rel=0.0, abs=1e-14)


@pytest.mark.parametrize("name", sorted(CAMPAIGNS))
def test_integrator_agrees_with_the_exact_closed_form(name: str):
    """Equation (20) and (21) of twindiverge.reference, which the integrator never uses.

    Exact in exact arithmetic; the tolerance is the measured rounding, not a
    fitted number.
    """
    campaign = CAMPAIGNS[name]
    integrated = simulate(NOMINAL, campaign).rate_rad_s
    closed_form = exact_campaign(NOMINAL, campaign)
    assert integrated == pytest.approx(closed_form, rel=5e-14, abs=0.0)


def test_integrator_converges_at_fourth_order():
    """Measured orders between 4.02 and 4.25 over steps from 0.4 s to 0.0125 s.

    The declared plant is too smooth to measure this on: its truncation error at
    a 0.1 s step is already below rounding. ORDER_PARAMETERS is a declared
    numerical exercise, documented in tests/conftest.py as not a claim about any
    wheel.
    """
    truth = float(
        exact_rate(ORDER_PARAMETERS, 1.0, ORDER_INITIAL_RATE, np.array([ORDER_DURATION]))[0]
    )
    steps = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)
    errors = [
        abs(float(simulate(ORDER_PARAMETERS, order_campaign(step)).rate_rad_s[-1]) - truth)
        for step in steps
    ]
    orders = [
        math.log2(errors[index] / errors[index + 1]) for index in range(len(steps) - 1)
    ]
    assert min(orders) > 3.8, orders
    assert max(orders) < 4.6, orders


@pytest.mark.parametrize("name", sorted(CAMPAIGNS))
def test_identity_a_the_rate_channel_cannot_see_the_torque_scale(name: str):
    """Equation (6): the five rate sensitivities sum to zero at every sample.

    Exact in exact arithmetic. The tolerance below is relative to the largest
    individual sensitivity, so it is a statement about cancellation and not
    about absolute size.
    """
    trajectory = simulate_with_sensitivities(NOMINAL, CAMPAIGNS[name])
    sensitivity = trajectory.rate_sensitivity
    assert sensitivity is not None
    residual = float(np.abs(sensitivity.sum(axis=1)).max())
    scale = float(np.abs(sensitivity).max())
    assert residual / scale < 1e-13, (residual, scale)


@pytest.mark.parametrize("name", sorted(CAMPAIGNS))
def test_identity_b_the_torque_channel_can(name: str):
    """Equation (7): the five torque sensitivities sum to the torque signal itself."""
    trajectory = simulate_with_sensitivities(NOMINAL, CAMPAIGNS[name])
    sensitivity = trajectory.torque_sensitivity
    assert sensitivity is not None
    assert sensitivity.sum(axis=1) == pytest.approx(
        trajectory.torque_nm, rel=1e-12, abs=1e-18
    )


def test_coast_campaign_has_an_exactly_unexcited_torque_constant():
    """The corollary to identity B: zero current, zero k_t sensitivity, exactly."""
    trajectory = simulate_with_sensitivities(NOMINAL, CAMPAIGNS["coast"])
    assert trajectory.rate_sensitivity is not None
    assert trajectory.torque_sensitivity is not None
    assert float(np.abs(trajectory.rate_sensitivity[:, 1]).max()) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )
    assert float(np.abs(trajectory.torque_sensitivity[:, 1]).max()) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


@pytest.mark.parametrize("index", range(5))
def test_forward_sensitivities_agree_with_central_differences(index: int):
    """The variational equation (3) against a finite difference of the state.

    A central difference at a declared step of 1e-5 in the logarithm has a
    truncation error of order 1e-10 relative, which is the tolerance below.
    """
    campaign = CAMPAIGNS["short"]
    analytic = simulate_with_sensitivities(NOMINAL, campaign).rate_sensitivity
    assert analytic is not None
    step = 1.0e-5
    offset = np.zeros(5)
    offset[index] = step
    forward = simulate(perturb(NOMINAL, offset), campaign).rate_rad_s
    backward = simulate(perturb(NOMINAL, -offset), campaign).rate_rad_s
    difference = (forward - backward) / (2.0 * step)
    scale = float(np.abs(analytic[:, index]).max())
    assert float(np.abs(difference - analytic[:, index]).max()) / scale < 1e-8


def test_perturbation_is_logarithmic():
    """exp(0.05) - 1 = 0.05127109637602412, not 0.05."""
    offset = np.zeros(5)
    offset[0] = 0.05
    moved = perturb(NOMINAL, offset)
    assert moved.inertia_kg_m2 / NOMINAL.inertia_kg_m2 - 1.0 == pytest.approx(
        0.05127109637602412, rel=1e-13, abs=0.0
    )


def test_a_trajectory_leaving_the_validity_window_raises():
    """Equation (1) carries the Coulomb torque as a constant; one sign only."""
    with pytest.raises(ValueError, match="validity window"):
        simulate(NOMINAL, Campaign("fast", 0.1, 51, 40.0, 5.0))


def test_initial_rate_outside_the_window_raises_before_a_single_step():
    with pytest.raises(ValueError, match="validity window"):
        simulate(NOMINAL, Campaign("low", 0.1, 11, 0.0, 0.0, initial_rate_rad_s=1.0))


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.032, 2.5e-3, 1.2e-5, 2.0e-8),
        (0.013, -0.032, 2.5e-3, 1.2e-5, 2.0e-8),
        (0.013, 0.032, 2.5e-3, 1.2e-5, 0.0),
    ],
)
def test_non_positive_parameters_raise(values: tuple[float, ...]):
    """The logarithmic parameterisation of equation (2) needs strict positivity."""
    with pytest.raises(ValueError, match="strictly positive"):
        WheelParameters(*values)


def test_non_finite_parameters_raise():
    with pytest.raises(ValueError, match="finite"):
        WheelParameters(float("nan"), 0.032, 2.5e-3, 1.2e-5, 2.0e-8)


def test_from_array_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match="expected 5 parameters"):
        WheelParameters.from_array(np.ones(4))


def test_perturb_rejects_the_wrong_shape_and_non_finite_values():
    with pytest.raises(ValueError, match="5 perturbation components"):
        perturb(NOMINAL, np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        perturb(NOMINAL, np.array([np.inf, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    ("period", "samples", "drive", "message"),
    [
        (0.0, 10, 0.0, "sample_period_s"),
        (0.1, 1, 0.0, "samples"),
        (0.1, 10, -1.0, "drive_duration_s must not be negative"),
        (0.1, 10, 0.15, "exact multiple"),
    ],
)
def test_campaign_validation(period: float, samples: int, drive: float, message: str):
    with pytest.raises(ValueError, match=message):
        Campaign("bad", period, samples, 1.0, drive)


def test_campaign_current_is_a_zero_order_hold():
    campaign = CAMPAIGNS["full"]
    current = campaign.current_a()
    assert float(current[0]) == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert float(current[399]) == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert float(current[400]) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert campaign.duration_s == pytest.approx(200.0, rel=1e-14, abs=0.0)


def test_simulate_and_simulate_with_sensitivities_agree_bit_for_bit():
    """One arithmetic path, not two. Bit-identical is asserted only here, where
    the two functions run literally the same step recurrence on the state."""
    plain = simulate(NOMINAL, CAMPAIGNS["short"])
    joint = simulate_with_sensitivities(NOMINAL, CAMPAIGNS["short"])
    assert np.array_equal(plain.rate_rad_s, joint.rate_rad_s)
    assert np.array_equal(plain.torque_nm, joint.torque_nm)


def test_rate_excursion_is_reported():
    trajectory = simulate(NOMINAL, CAMPAIGNS["full"])
    assert trajectory.rate_excursion_rad_s == pytest.approx(
        float(trajectory.rate_rad_s.max() - trajectory.rate_rad_s.min()), rel=0.0, abs=0.0
    )
    assert OMEGA_MIN < float(trajectory.rate_rad_s.min())
    assert float(trajectory.rate_rad_s.max()) < OMEGA_MAX


def test_exact_rate_rejects_negative_times_and_a_pole():
    with pytest.raises(ValueError, match="must not be negative"):
        exact_rate(NOMINAL, 0.0, 100.0, np.array([-1.0]))
    with pytest.raises(ValueError, match="pole"):
        exact_rate(NOMINAL, 0.0, 100.0, np.array([1.0e7]))
