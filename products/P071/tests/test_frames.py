"""Radial / along-track / cross-track frame construction."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.frames import covariance_from_rtn_sigmas, rtn_basis


def test_basis_is_orthonormal_and_right_handed():
    basis = rtn_basis([7078137.0, 0.0, 0.0], [10.0, 7504.0, 0.0])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    assert float(np.linalg.det(basis)) == pytest.approx(1.0, rel=1e-13)


def test_basis_for_a_circular_equatorial_orbit_is_the_identity_permutation():
    # Position along x, velocity along y: radial = x, along-track = y,
    # cross-track = x cross y = z.
    basis = rtn_basis([7078137.0, 0.0, 0.0], [0.0, 7504.0, 0.0])
    np.testing.assert_allclose(basis, np.eye(3), atol=1e-14)


def test_covariance_from_rtn_is_symmetric_positive_definite_with_the_right_spectrum():
    covariance = covariance_from_rtn_sigmas(
        [7078137.0, 0.0, 0.0], [0.0, 7504.0, 0.0], [20.0, 100.0, 20.0]
    )
    np.testing.assert_allclose(covariance, covariance.T, atol=0.0)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
    np.testing.assert_allclose(eigenvalues, [400.0, 400.0, 10000.0], rtol=1e-12)


def test_covariance_from_rtn_is_rotation_covariant():
    position = np.array([7078137.0, 0.0, 0.0])
    velocity = np.array([0.0, 7504.0, 0.0])
    angle = 0.9
    rotation = np.array(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    base = covariance_from_rtn_sigmas(position, velocity, [20.0, 100.0, 20.0])
    rotated = covariance_from_rtn_sigmas(
        rotation @ position, rotation @ velocity, [20.0, 100.0, 20.0]
    )
    np.testing.assert_allclose(rotated, rotation @ base @ rotation.T, rtol=1e-10, atol=1e-8)


def test_correlation_changes_the_spectrum_but_keeps_the_trace():
    position = [7078137.0, 0.0, 0.0]
    velocity = [0.0, 7504.0, 0.0]
    plain = covariance_from_rtn_sigmas(position, velocity, [20.0, 100.0, 20.0])
    correlated = covariance_from_rtn_sigmas(
        position, velocity, [20.0, 100.0, 20.0], correlation_ra=0.7
    )
    assert float(np.trace(correlated)) == pytest.approx(float(np.trace(plain)), rel=1e-12)
    assert np.linalg.eigvalsh(correlated).max() > np.linalg.eigvalsh(plain).max()
    assert np.linalg.eigvalsh(correlated).min() > 0.0


@pytest.mark.parametrize(
    "position, velocity, sigmas, correlation, message",
    [
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], 0.0, "non-zero"),
        ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, "perpendicular"),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0], 0.0, "3 components"),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], 0.0, "strictly positive"),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], 1.0, "correlation_ra"),
    ],
)
def test_rejects_bad_input(position, velocity, sigmas, correlation, message):
    with pytest.raises(ValueError, match=message):
        covariance_from_rtn_sigmas(position, velocity, sigmas, correlation_ra=correlation)


def test_rtn_basis_rejects_short_vectors():
    with pytest.raises(ValueError, match="3 components"):
        rtn_basis([1.0, 0.0], [0.0, 1.0, 0.0])
