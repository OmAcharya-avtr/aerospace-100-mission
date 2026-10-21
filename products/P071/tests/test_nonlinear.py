"""Measured breakdown of the rectilinear-relative-motion assumption."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conjprob.analytic import pc_foster
from conjprob.geometry import encounter_plane_basis, principal_frame, project_covariance
from conjprob.nonlinear import (
    EARTH_MU_M3_S2,
    curvature_parameter,
    curvature_sweep,
    first_crossing,
    gravity_gradient_acceleration_scale,
    window_sweep,
)

COVARIANCE = np.diag([54.0**2, 300.0**2, 80.0**2])
MISS = np.array([120.0, 30.0, 0.0])
VELOCITY = np.array([0.0, 0.0, 5740.0])
RADIUS = 12.0


def _frame():
    basis = encounter_plane_basis(VELOCITY)
    return principal_frame(project_covariance(COVARIANCE, basis), basis.T @ MISS)


def test_gravity_gradient_scale_hand_calculated():
    # mu |dr| / r^3 with mu = 3.986004418e14 m^3/s^2, r = 7078137 m,
    # |dr| = 300 m gives 3.3721131400089e-4 m/s^2.
    value = gravity_gradient_acceleration_scale(7078137.0, 300.0)
    expected = EARTH_MU_M3_S2 * 300.0 / 7078137.0**3
    assert value == pytest.approx(expected, rel=1e-15)
    assert value == pytest.approx(3.372113140008893e-4, rel=1e-12)


def test_gravity_gradient_scale_rejects_bad_input():
    with pytest.raises(ValueError, match="geocentric_radius_m"):
        gravity_gradient_acceleration_scale(0.0, 1.0)
    with pytest.raises(ValueError, match="separation_m"):
        gravity_gradient_acceleration_scale(7e6, -1.0)


def test_curvature_parameter_hand_calculated():
    # |a| T^2 / (2 sigma_eff) = 1e-4 * 100^2 / (2 * 200) = 0.0025.
    assert curvature_parameter(1e-4, 100.0, 200.0) == pytest.approx(0.0025, rel=1e-15)


def test_curvature_parameter_rejects_bad_input():
    with pytest.raises(ValueError, match="sigma_effective_m"):
        curvature_parameter(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="window_s"):
        curvature_parameter(1.0, -1.0, 1.0)


def test_first_crossing_linear_interpolation():
    # y goes 0.0, 0.5, 2.0 at x = 1, 2, 3, so |y| crosses 1.0 between x = 2
    # and x = 3 at 2 + (1.0 - 0.5) / (2.0 - 0.5) = 2.3333333333333335.
    assert first_crossing([1.0, 2.0, 3.0], [0.0, 0.5, 2.0], 1.0) == pytest.approx(
        2.0 + 0.5 / 1.5, rel=1e-13
    )


def test_first_crossing_returns_nan_when_never_crossed():
    assert math.isnan(first_crossing([1.0, 2.0, 3.0], [0.0, 0.1, 0.2], 1.0))


def test_first_crossing_returns_the_first_abscissa_when_already_above():
    assert first_crossing([5.0, 6.0], [2.0, 3.0], 1.0) == pytest.approx(5.0, rel=0.0)


def test_first_crossing_rejects_bad_input():
    with pytest.raises(ValueError, match="threshold"):
        first_crossing([1.0, 2.0], [1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="equal length"):
        first_crossing([1.0, 2.0], [1.0], 1.0)


def test_window_sweep_ratios_increase_towards_one():
    frame = _frame()
    unbounded, points = window_sweep(
        COVARIANCE,
        MISS,
        VELOCITY,
        RADIUS,
        [1e-4, 1e-3, 1e-2, 1.0],
        sigma_effective_m=math.sqrt(frame.sigma_x_m * frame.sigma_y_m),
        n_samples=200_000,
        seed=21,
    )
    ratios = [point.ratio_to_unbounded for point in points]
    assert all(0.0 <= ratio <= 1.0 + 1e-12 for ratio in ratios)
    assert ratios[-1] == pytest.approx(1.0, rel=1e-12)
    assert ratios[0] < ratios[-1]
    assert points[0].window_over_crossing_time < points[-1].window_over_crossing_time
    assert unbounded.hits > 0


def test_curvature_sweep_error_grows_with_the_curvature_parameter():
    frame = _frame()
    sigma_eff = math.sqrt(frame.sigma_x_m * frame.sigma_y_m)
    reference = pc_foster(
        frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, RADIUS
    )
    points = curvature_sweep(
        COVARIANCE,
        MISS,
        VELOCITY,
        RADIUS,
        [0.0, 1e6, 1e7],
        window_s=0.2,
        pc_linear_2d=reference,
        sigma_effective_m=sigma_eff,
        n_samples=40_000,
        seed=31,
        time_nodes=65,
        refinements=3,
    )
    assert points[0].curvature_parameter == pytest.approx(0.0, abs=0.0)
    assert points[-1].curvature_parameter > points[1].curvature_parameter
    assert abs(points[-1].relative_error) > abs(points[0].relative_error)


def test_curvature_sweep_rejects_a_zero_miss_without_a_direction():
    with pytest.raises(ValueError, match="acceleration_direction"):
        curvature_sweep(
            COVARIANCE,
            [0.0, 0.0, 0.0],
            VELOCITY,
            RADIUS,
            [1.0],
            window_s=1.0,
            pc_linear_2d=1e-3,
            sigma_effective_m=100.0,
            n_samples=10,
        )


def test_curvature_sweep_rejects_a_degenerate_direction():
    with pytest.raises(ValueError, match="non-zero 3-vector"):
        curvature_sweep(
            COVARIANCE,
            MISS,
            VELOCITY,
            RADIUS,
            [1.0],
            window_s=1.0,
            pc_linear_2d=1e-3,
            sigma_effective_m=100.0,
            acceleration_direction=[0.0, 0.0, 0.0],
            n_samples=10,
        )
