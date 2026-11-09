"""Telemetry sets: the design matrix, the residual, and the noise draw."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.plant import CAMPAIGNS, NOMINAL, simulate_with_sensitivities
from twindiverge.telemetry import (
    RATE_SIGMA_RAD_S,
    TELEMETRY_SETS,
    TORQUE_SIGMA_NM,
    TelemetrySet,
    design_matrix,
    measure,
    residual_vector,
)


@pytest.mark.parametrize("name", sorted(TELEMETRY_SETS))
def test_design_matrix_shape_matches_the_declared_row_count(name: str):
    telemetry = TELEMETRY_SETS[name]
    matrix, _ = design_matrix(telemetry, NOMINAL)
    assert matrix.shape == (telemetry.rows, 5)
    assert telemetry.rows == len(telemetry.channels) * telemetry.campaign.samples


def test_design_matrix_rows_are_scaled_by_the_declared_noise():
    """Equation (8): the rate block is the sensitivity divided by 0.02 rad/s."""
    telemetry = TELEMETRY_SETS["rate_short"]
    matrix, trajectory = design_matrix(telemetry, NOMINAL)
    assert trajectory.rate_sensitivity is not None
    assert matrix == pytest.approx(
        trajectory.rate_sensitivity / RATE_SIGMA_RAD_S, rel=1e-15, abs=0.0
    )


def test_two_channel_design_matrix_stacks_in_declared_channel_order():
    telemetry = TELEMETRY_SETS["rate_torque_short"]
    matrix, trajectory = design_matrix(telemetry, NOMINAL)
    width = telemetry.campaign.samples
    assert trajectory.rate_sensitivity is not None
    assert trajectory.torque_sensitivity is not None
    assert matrix[:width] == pytest.approx(
        trajectory.rate_sensitivity / RATE_SIGMA_RAD_S, rel=1e-15, abs=0.0
    )
    assert matrix[width:] == pytest.approx(
        trajectory.torque_sensitivity / TORQUE_SIGMA_NM, rel=1e-15, abs=0.0
    )


def test_design_matrix_refuses_a_trajectory_without_sensitivities():
    from twindiverge.plant import simulate

    telemetry = TELEMETRY_SETS["rate_short"]
    with pytest.raises(ValueError, match="carrying sensitivities"):
        design_matrix(telemetry, NOMINAL, trajectory=simulate(NOMINAL, telemetry.campaign))


def test_design_matrix_refuses_a_trajectory_from_another_campaign():
    telemetry = TELEMETRY_SETS["rate_short"]
    other = simulate_with_sensitivities(NOMINAL, CAMPAIGNS["coast"])
    with pytest.raises(ValueError, match="produced for campaign"):
        design_matrix(telemetry, NOMINAL, trajectory=other)


def test_measure_is_a_function_of_the_seed_alone():
    telemetry = TELEMETRY_SETS["rate_short"]
    first, _ = measure(telemetry, NOMINAL, seed=7)
    second, _ = measure(telemetry, NOMINAL, seed=7)
    third, _ = measure(telemetry, NOMINAL, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_measure_rejects_a_negative_seed():
    with pytest.raises(ValueError, match="must not be negative"):
        measure(TELEMETRY_SETS["rate_short"], NOMINAL, seed=-1)


def test_residual_of_the_truth_is_standard_normal_in_scale():
    """The scaled residual has unit variance by construction, so its chi-square
    is close to its row count. 201 rows: a chi-square on 201 degrees of freedom
    has standard deviation sqrt(402) = 20.05, and the tolerance below is four of
    those, the declared control sigma."""
    telemetry = TELEMETRY_SETS["rate_short"]
    observed, _ = measure(telemetry, NOMINAL, seed=11)
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    residual = residual_vector(telemetry, observed, twin)
    chi2 = float(residual @ residual)
    assert abs(chi2 - telemetry.rows) < 4.0 * np.sqrt(2.0 * telemetry.rows)


def test_residual_rejects_the_wrong_shape_and_the_wrong_campaign():
    telemetry = TELEMETRY_SETS["rate_short"]
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    with pytest.raises(ValueError, match="rows"):
        residual_vector(telemetry, np.zeros(3), twin)
    other = simulate_with_sensitivities(NOMINAL, CAMPAIGNS["coast"])
    with pytest.raises(ValueError, match="simulated for campaign"):
        residual_vector(telemetry, np.zeros(telemetry.rows), other)


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        ((), "at least one channel"),
        (("rate", "rate"), "distinct"),
        (("torque", "temperature"), "unknown channel"),
    ],
)
def test_telemetry_set_validation(channels: tuple[str, ...], message: str):
    with pytest.raises(ValueError, match=message):
        TelemetrySet("bad", CAMPAIGNS["short"], channels)


def test_a_non_positive_noise_level_raises():
    with pytest.raises(ValueError, match="strictly positive noise"):
        TelemetrySet("bad", CAMPAIGNS["short"], ("rate",), {"rate": 0.0})
