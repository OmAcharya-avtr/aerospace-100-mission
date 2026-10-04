"""Geometry: range, range-rate, range-acceleration, finite differences."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dopplerkit.geometry import (
    RANGE_RATE_CONVENTION,
    State,
    range_acceleration_mps2,
    range_rate_finite_difference_mps,
    range_rate_mps,
    relative_position_m,
    slant_range_m,
)


def test_state_validates_shape():
    with pytest.raises(ValueError, match="position_m must have shape"):
        State(position_m=[1.0, 2.0], velocity_mps=np.zeros(3))
    with pytest.raises(ValueError, match="velocity_mps must have shape"):
        State(position_m=np.zeros(3), velocity_mps=[0.0])
    with pytest.raises(ValueError, match="acceleration_mps2 must have shape"):
        State(position_m=np.zeros(3), velocity_mps=np.zeros(3), acceleration_mps2=[1.0, 2.0])


def test_state_rejects_non_finite():
    with pytest.raises(ValueError, match="must be finite"):
        State(position_m=[np.nan, 0.0, 0.0], velocity_mps=np.zeros(3))
    with pytest.raises(ValueError, match="must be finite"):
        State(position_m=np.zeros(3), velocity_mps=[np.inf, 0.0, 0.0])


def test_state_label_must_be_str():
    with pytest.raises(TypeError, match="label must be str"):
        State(position_m=np.zeros(3), velocity_mps=np.zeros(3), label=3)


def test_state_default_acceleration_is_zero_and_speed():
    s = State(position_m=[1.0, 0.0, 0.0], velocity_mps=[3.0, 4.0, 0.0])
    assert np.array_equal(s.acceleration_mps2, np.zeros(3))
    assert s.speed_mps == pytest.approx(5.0)


def test_hand_calculated_range_and_range_rate():
    # Hand calculation. Observer at origin, target at (3, 4, 0) km -> range 5 km.
    # Target velocity (6, 8, 0) m/s is exactly along +rho_hat = (0.6, 0.8, 0),
    # so rho_dot = |v| = 10 m/s, POSITIVE because it is receding.
    obs = State(position_m=[0.0, 0.0, 0.0], velocity_mps=[0.0, 0.0, 0.0])
    tgt = State(position_m=[3000.0, 4000.0, 0.0], velocity_mps=[6.0, 8.0, 0.0])
    assert slant_range_m(obs, tgt) == pytest.approx(5000.0, abs=0.0)
    assert range_rate_mps(obs, tgt) == pytest.approx(10.0, abs=1e-12)


def test_hand_calculated_tangential_velocity_gives_zero_range_rate():
    # Velocity (-8, 6, 0) is perpendicular to rho_hat = (0.6, 0.8, 0):
    # dot product = -4.8 + 4.8 = 0 exactly. Pure tangential motion, no Doppler.
    obs = State(position_m=[0.0, 0.0, 0.0], velocity_mps=[0.0, 0.0, 0.0])
    tgt = State(position_m=[3000.0, 4000.0, 0.0], velocity_mps=[-8.0, 6.0, 0.0])
    assert range_rate_mps(obs, tgt) == pytest.approx(0.0, abs=1e-12)


def test_hand_calculated_range_acceleration_pure_tangential():
    # rho = 5000 m, v_rel = 10 m/s purely tangential, a_rel = 0.
    # rho_ddot = (|v|^2 + 0 - 0) / rho = 100 / 5000 = 0.02 m/s^2 exactly.
    obs = State(position_m=np.zeros(3), velocity_mps=np.zeros(3))
    tgt = State(position_m=[3000.0, 4000.0, 0.0], velocity_mps=[-8.0, 6.0, 0.0])
    assert range_acceleration_mps2(obs, tgt) == pytest.approx(0.02, abs=1e-15)


def test_hand_calculated_range_acceleration_pure_radial():
    # Radial motion only: |v|^2 == rho_dot^2 so the kinematic term cancels
    # exactly and rho_ddot = rho . a_rel / rho = |a| = 2.0 m/s^2.
    obs = State(position_m=np.zeros(3), velocity_mps=np.zeros(3))
    tgt = State(
        position_m=[3000.0, 4000.0, 0.0],
        velocity_mps=[6.0, 8.0, 0.0],
        acceleration_mps2=[1.2, 1.6, 0.0],
    )
    assert range_acceleration_mps2(obs, tgt) == pytest.approx(2.0, abs=1e-12)


def test_relative_position_points_observer_to_target():
    obs = State(position_m=[1.0, 2.0, 3.0], velocity_mps=np.zeros(3))
    tgt = State(position_m=[5.0, 7.0, 9.0], velocity_mps=np.zeros(3))
    assert np.array_equal(relative_position_m(obs, tgt), np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(relative_position_m(tgt, obs), np.array([-4.0, -5.0, -6.0]))


def test_coincident_positions_raise():
    s = State(position_m=[1.0, 1.0, 1.0], velocity_mps=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="coincide"):
        slant_range_m(s, s)
    with pytest.raises(ValueError, match="coincide"):
        range_rate_mps(s, s)
    with pytest.raises(ValueError, match="coincide"):
        range_acceleration_mps2(s, s)


def test_finite_difference_matches_analytic_derivative():
    # rho(t) = sqrt(1 + t^2) metres, rho'(t) = t / sqrt(1 + t^2).
    def rho(t: float) -> float:
        return math.sqrt(1.0 + t * t)

    for t in (-2.0, -0.5, 0.0, 0.3, 1.7):
        exact = t / math.sqrt(1.0 + t * t)
        assert range_rate_finite_difference_mps(rho, t, 1e-5) == pytest.approx(exact, abs=1e-9)


def test_finite_difference_rejects_bad_step():
    with pytest.raises(ValueError, match="step_s must be a positive finite"):
        range_rate_finite_difference_mps(lambda t: t, 0.0, 0.0)
    with pytest.raises(ValueError, match="step_s must be a positive finite"):
        range_rate_finite_difference_mps(lambda t: t, 0.0, -1.0)
    with pytest.raises(ValueError, match="step_s must be a positive finite"):
        range_rate_finite_difference_mps(lambda t: t, 0.0, float("nan"))


def test_convention_string_names_receding_as_positive():
    assert "receding" in RANGE_RATE_CONVENTION
    assert "separating" in RANGE_RATE_CONVENTION
