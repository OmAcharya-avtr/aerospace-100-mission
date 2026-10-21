"""The independent 2-D quadrature reference."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.analytic import pc_circular_closed_form, pc_foster
from conjprob.quadrature import pc_quadrature


def test_matches_rayleigh_closed_form():
    value, abserr = pc_quadrature(np.eye(2) * 100.0**2, [0.0, 0.0], 20.0)
    assert value == pytest.approx(pc_circular_closed_form(20.0, 100.0), rel=1e-10)
    assert abserr < 1e-12


def test_invariant_under_rotation_of_the_in_plane_basis():
    covariance = np.diag([300.0**2, 54.0**2])
    miss = np.array([120.0, 30.0])
    base, _ = pc_quadrature(covariance, miss, 12.0)
    for angle in (0.3, 1.1, 2.7, 5.0):
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated, _ = pc_quadrature(
            rotation @ covariance @ rotation.T, rotation @ miss, 12.0
        )
        assert rotated == pytest.approx(base, rel=1e-9)


def test_handles_a_correlated_covariance_without_diagonalising():
    covariance = np.array([[2500.0, 1200.0], [1200.0, 900.0]])
    miss = np.array([40.0, -15.0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    sigma = np.sqrt(eigenvalues[order])
    components = eigenvectors[:, order].T @ miss
    expected = pc_foster(sigma[0], sigma[1], components[0], components[1], 5.0)
    value, abserr = pc_quadrature(covariance, miss, 5.0)
    assert value == pytest.approx(expected, rel=1e-8, abs=10.0 * abserr)


def test_zero_radius_gives_zero():
    assert pc_quadrature(np.eye(2), [0.0, 0.0], 0.0) == (0.0, 0.0)


def test_large_radius_approaches_one():
    value, _ = pc_quadrature(np.eye(2) * 100.0, [0.0, 0.0], 1000.0)
    assert value == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize(
    "covariance, miss, radius, message",
    [
        (np.eye(3), [0.0, 0.0], 1.0, r"shape \(2, 2\)"),
        (np.diag([1.0, 0.0]), [0.0, 0.0], 1.0, "positive definite"),
        (np.eye(2), [0.0, 0.0, 0.0], 1.0, "2 components"),
        (np.eye(2), [0.0, 0.0], -1.0, "non-negative"),
        (np.full((2, 2), np.nan), [0.0, 0.0], 1.0, "finite"),
    ],
)
def test_rejects_bad_input(covariance, miss, radius, message):
    with pytest.raises(ValueError, match=message):
        pc_quadrature(covariance, miss, radius)
