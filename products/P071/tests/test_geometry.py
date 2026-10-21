"""Encounter geometry: known answers, invariances and input validation."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.geometry import (
    ObjectState,
    encounter_geometry,
    encounter_plane_basis,
    mahalanobis_distance,
    miss_vector,
    principal_frame,
    project_covariance,
    relative_state,
    time_of_closest_approach,
)


def test_tca_hand_calculated():
    # r = (100, 0, 0) m, v = (-10, 0, 0) m/s. The objects close head on at
    # 10 m/s from 100 m, so closest approach is at t = +10 s and the miss is 0.
    assert time_of_closest_approach([100.0, 0.0, 0.0], [-10.0, 0.0, 0.0]) == pytest.approx(
        10.0, rel=1e-12
    )
    assert np.linalg.norm(miss_vector([100.0, 0.0, 0.0], [-10.0, 0.0, 0.0])) == pytest.approx(
        0.0, abs=1e-9
    )


def test_tca_perpendicular_geometry_is_zero():
    # r perpendicular to v means the epoch already is the closest approach.
    assert time_of_closest_approach([0.0, 50.0, 0.0], [7000.0, 0.0, 0.0]) == pytest.approx(
        0.0, abs=1e-15
    )


def test_miss_vector_is_perpendicular_to_relative_velocity():
    r = np.array([1234.0, -567.0, 89.0])
    v = np.array([-3000.0, 4000.0, 500.0])
    assert float(miss_vector(r, v) @ v) == pytest.approx(0.0, abs=1e-6)


def test_miss_distance_hand_calculated():
    # r = (0, 300, 400) m with v along x: the epoch is the TCA and the miss is
    # the 3-4-5 hypotenuse, 500 m exactly in exact arithmetic.
    assert np.linalg.norm(miss_vector([0.0, 300.0, 400.0], [8000.0, 0.0, 0.0])) == pytest.approx(
        500.0, rel=1e-12
    )


def test_encounter_plane_basis_is_orthonormal_and_perpendicular():
    v = np.array([1.0, -2.0, 3.0]) * 1000.0
    basis = encounter_plane_basis(v)
    assert basis.shape == (3, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(basis.T @ v, np.zeros(2), atol=1e-9)


def test_projected_covariance_of_identity_is_identity():
    basis = encounter_plane_basis([0.0, 0.0, 7000.0])
    np.testing.assert_allclose(project_covariance(np.eye(3), basis), np.eye(2), atol=1e-14)


def test_principal_frame_orders_sigma_descending_and_preserves_trace():
    covariance = np.array([[400.0, 120.0], [120.0, 2500.0]])
    frame = principal_frame(covariance, [10.0, -20.0])
    assert frame.sigma_x_m >= frame.sigma_y_m
    assert frame.sigma_x_m**2 + frame.sigma_y_m**2 == pytest.approx(
        float(np.trace(covariance)), rel=1e-12
    )
    assert frame.aspect_ratio >= 1.0


def test_principal_frame_preserves_miss_distance():
    covariance = np.array([[900.0, -200.0], [-200.0, 1600.0]])
    miss = np.array([30.0, 40.0])
    frame = principal_frame(covariance, miss)
    assert frame.miss_distance_m == pytest.approx(50.0, rel=1e-12)


def test_mahalanobis_matches_explicit_quadratic_form():
    covariance = np.array([[900.0, 150.0], [150.0, 1600.0]])
    miss = np.array([25.0, -40.0])
    expected = float(np.sqrt(miss @ np.linalg.inv(covariance) @ miss))
    assert mahalanobis_distance(covariance, miss) == pytest.approx(expected, rel=1e-12)


def test_relative_state_sums_covariances(leo_primary, leo_secondary):
    _, _, covariance = relative_state(leo_primary, leo_secondary)
    np.testing.assert_allclose(
        covariance, leo_primary.covariance_m2 + leo_secondary.covariance_m2, atol=0.0
    )


def test_encounter_geometry_fields_are_consistent(leo_primary, leo_secondary):
    geometry = encounter_geometry(leo_primary, leo_secondary)
    assert geometry.miss_distance_m == pytest.approx(
        float(np.linalg.norm(geometry.miss_vector_m)), rel=1e-12
    )
    assert geometry.frame.miss_distance_m == pytest.approx(geometry.miss_distance_m, rel=1e-9)
    assert geometry.sigma_effective_m == pytest.approx(
        float(np.sqrt(geometry.frame.sigma_x_m * geometry.frame.sigma_y_m)), rel=1e-12
    )
    assert geometry.crossing_time_s == pytest.approx(
        geometry.sigma_effective_m / geometry.relative_speed_mps, rel=1e-12
    )


def test_encounter_geometry_mahalanobis_matches_module_function(leo_primary, leo_secondary):
    geometry = encounter_geometry(leo_primary, leo_secondary)
    assert geometry.mahalanobis == pytest.approx(
        mahalanobis_distance(geometry.covariance_2d_m2, geometry.miss_2d_m), rel=1e-12
    )


@pytest.mark.parametrize(
    "position, velocity, covariance, message",
    [
        ([0.0, 0.0], [1.0, 0.0, 0.0], np.eye(3), "position_m"),
        ([0.0, 0.0, 0.0], [1.0, 0.0], np.eye(3), "velocity_mps"),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], np.eye(2), "shape"),
        ([np.nan, 0.0, 0.0], [1.0, 0.0, 0.0], np.eye(3), "finite"),
    ],
)
def test_object_state_rejects_bad_input(position, velocity, covariance, message):
    with pytest.raises(ValueError, match=message):
        ObjectState(position_m=position, velocity_mps=velocity, covariance_m2=covariance)


def test_object_state_rejects_asymmetric_covariance():
    covariance = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        ObjectState(
            position_m=[1.0, 0.0, 0.0],
            velocity_mps=[0.0, 1.0, 0.0],
            covariance_m2=covariance,
        )


def test_object_state_rejects_indefinite_covariance():
    covariance = np.diag([1.0, 1.0, -4.0])
    with pytest.raises(ValueError, match="positive semi-definite"):
        ObjectState(
            position_m=[1.0, 0.0, 0.0],
            velocity_mps=[0.0, 1.0, 0.0],
            covariance_m2=covariance,
        )


def test_zero_relative_velocity_is_rejected():
    with pytest.raises(ValueError, match="relative velocity is zero"):
        time_of_closest_approach([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="relative velocity is zero"):
        encounter_plane_basis([0.0, 0.0, 0.0])


def test_singular_projected_covariance_is_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        principal_frame(np.array([[1.0, 0.0], [0.0, 0.0]]), [1.0, 1.0])


def test_project_covariance_rejects_wrong_basis_shape():
    with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
        project_covariance(np.eye(3), np.eye(3))
