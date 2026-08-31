"""Attitude covariance: closed forms, orderings, Monte Carlo agreement."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import synthetic_problem, transverse_noise, well_conditioned_reference
from wahbakit import (
    DegenerateObservationsError,
    VectorObservations,
    attitude_covariance,
    attitude_error_vector,
    covariance_axis_sigmas_deg,
    dcm_from_quat,
    optimal_covariance,
    q_method,
    triad,
    triad_covariance,
)


def test_optimal_covariance_known_answer_two_orthogonal_observations():
    # b1 = x, b2 = y, equal sigma: F = sigma^-2 diag(1, 1, 2),
    # so P = sigma^2 diag(1, 1, 1/2) by hand.
    sigma = 2e-3
    obs = VectorObservations(
        [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]], sigmas=[sigma, sigma]
    )
    expected = sigma**2 * np.diag([1.0, 1.0, 0.5])
    assert np.allclose(optimal_covariance(obs), expected, atol=1e-18)


def test_optimal_covariance_known_answer_three_orthogonal_observations():
    # Three orthogonal axes, equal sigma: F = 2 sigma^-2 I, P = sigma^2 I / 2.
    sigma = 1e-3
    obs = VectorObservations(np.eye(3), np.eye(3), sigmas=np.full(3, sigma))
    assert np.allclose(optimal_covariance(obs), 0.5 * sigma**2 * np.eye(3), atol=1e-20)


def test_optimal_covariance_is_symmetric_positive_definite():
    rng = np.random.default_rng(10)
    obs, _ = synthetic_problem(rng, n=5, sigma=1e-3)
    covariance = optimal_covariance(obs)
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_covariance_scales_as_sigma_squared():
    reference = well_conditioned_reference()
    dcm = dcm_from_quat([0.5, 0.5, 0.5, 0.5])
    body = reference @ dcm.T
    base = optimal_covariance(VectorObservations(body, reference, sigmas=np.full(4, 1e-3)))
    scaled = optimal_covariance(VectorObservations(body, reference, sigmas=np.full(4, 3e-3)))
    assert np.allclose(scaled, 9.0 * base, rtol=1e-12)


def test_triad_covariance_known_answer_orthogonal_pair():
    # eta = 90 deg: c = 0, s = 1, so P' = diag(sigma2^2, sigma1^2, sigma1^2)
    # in the triad basis [b1, b1 x b2, b1 x (b1 x b2)], with no cross term.
    sigma1, sigma2 = 1e-3, 4e-3
    obs = VectorObservations(
        [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]], sigmas=[sigma1, sigma2]
    )
    covariance = triad_covariance(obs)
    assert np.allclose(np.diag(covariance), [sigma2**2, sigma1**2, sigma1**2], atol=1e-20)
    assert np.allclose(covariance - np.diag(np.diag(covariance)), 0.0, atol=1e-20)


def test_triad_covariance_grows_as_one_over_sin_squared():
    # Var about the primary direction = (sigma1^2 c^2 + sigma2^2) / s^2.
    sigma = 1e-3
    variances = []
    for separation_deg in (90.0, 30.0, 10.0, 3.0):
        eta = np.radians(separation_deg)
        vectors = np.array([[1.0, 0.0, 0.0], [np.cos(eta), np.sin(eta), 0.0]])
        obs = VectorObservations(vectors, vectors, sigmas=[sigma, sigma])
        covariance = triad_covariance(obs)
        expected = (sigma**2 * np.cos(eta) ** 2 + sigma**2) / np.sin(eta) ** 2
        variances.append(float(covariance[0, 0]))
        assert covariance[0, 0] == pytest.approx(expected, rel=1e-12)
    assert variances == sorted(variances)


def test_triad_covariance_is_never_smaller_than_the_optimum():
    rng = np.random.default_rng(12)
    for _ in range(30):
        obs, _ = synthetic_problem(rng, n=2, sigma=1e-3)
        difference = triad_covariance(obs) - optimal_covariance(obs)
        assert np.min(np.linalg.eigvalsh(difference)) > -1e-18


def test_triad_covariance_approaches_the_optimum_as_the_primary_gets_exact():
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gaps = []
    for sigma1 in (1e-3, 1e-4, 1e-5, 1e-6):
        obs = VectorObservations(reference, reference, sigmas=[sigma1, 1e-3])
        gaps.append(float(np.max(triad_covariance(obs) - optimal_covariance(obs))))
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-12


@pytest.mark.parametrize("method", ["optimal", "q-method", "quest", "olae"])
def test_attitude_covariance_dispatch_returns_the_optimal_form(method):
    rng = np.random.default_rng(13)
    obs, _ = synthetic_problem(rng, n=4, sigma=1e-3)
    assert np.allclose(attitude_covariance(obs, method), optimal_covariance(obs))


def test_attitude_covariance_triad_dispatch():
    rng = np.random.default_rng(14)
    obs, _ = synthetic_problem(rng, n=2, sigma=1e-3)
    assert np.allclose(attitude_covariance(obs, "triad"), triad_covariance(obs))
    assert np.allclose(
        attitude_covariance(obs, "triad", primary=1), triad_covariance(obs, primary=1)
    )


def test_attitude_covariance_rejects_unknown_method():
    rng = np.random.default_rng(15)
    obs, _ = synthetic_problem(rng, n=3, sigma=1e-3)
    with pytest.raises(ValueError, match="unknown method"):
        attitude_covariance(obs, "kalman")


def test_covariance_requires_sigmas():
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="requires per-observation sigmas"):
        optimal_covariance(obs)


def test_covariance_rejects_degenerate_geometry():
    vectors = np.array([[1.0, 0.0, 0.0], [1.0, 1e-5, 0.0]])
    obs = VectorObservations(vectors, vectors, sigmas=[1e-3, 1e-3])
    with pytest.raises(DegenerateObservationsError):
        optimal_covariance(obs)


def test_triad_covariance_rejects_wrong_observation_count_and_primary():
    rng = np.random.default_rng(16)
    obs3, _ = synthetic_problem(rng, n=3, sigma=1e-3)
    with pytest.raises(ValueError, match="exactly 2 observations"):
        triad_covariance(obs3)
    obs2, _ = synthetic_problem(rng, n=2, sigma=1e-3)
    with pytest.raises(ValueError, match="primary must be 0 or 1"):
        triad_covariance(obs2, primary=5)


def test_covariance_axis_sigmas_deg():
    covariance = np.diag([np.radians(0.1) ** 2, np.radians(0.2) ** 2, np.radians(0.3) ** 2])
    assert np.allclose(covariance_axis_sigmas_deg(covariance), [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="3, 3"):
        covariance_axis_sigmas_deg(np.eye(2))


def _monte_carlo_covariance(solver, reference, dcm, sigmas, trials, seed):
    rng = np.random.default_rng(seed)
    true_body = reference @ dcm.T
    errors = np.empty((trials, 3))
    for k in range(trials):
        obs = VectorObservations(
            transverse_noise(true_body, sigmas, rng), reference, sigmas=sigmas
        )
        errors[k] = attitude_error_vector(solver(obs).dcm, dcm)
    return errors.T @ errors / trials


def test_optimal_covariance_matches_monte_carlo():
    reference = well_conditioned_reference()
    dcm = dcm_from_quat([0.3, -0.7, 1.1, 0.2])
    sigmas = np.array([1e-3, 2e-3, 5e-3, 1e-3])
    trials = 4000
    empirical = _monte_carlo_covariance(q_method, reference, dcm, sigmas, trials, 424242)
    analytic = optimal_covariance(VectorObservations(reference @ dcm.T, reference, sigmas=sigmas))
    # Sampling error on a covariance entry is ~sqrt(2 / trials) = 2.2 % here;
    # 12 % is a loose but non-vacuous gate that a wrong formula fails outright.
    relative = np.abs(empirical - analytic) / np.max(np.abs(analytic))
    assert np.max(relative) < 0.12


def test_triad_covariance_matches_monte_carlo():
    reference = well_conditioned_reference()[:2]
    dcm = dcm_from_quat([0.3, -0.7, 1.1, 0.2])
    sigmas = np.array([1e-3, 5e-3])
    trials = 4000
    empirical = _monte_carlo_covariance(triad, reference, dcm, sigmas, trials, 515151)
    analytic = triad_covariance(VectorObservations(reference @ dcm.T, reference, sigmas=sigmas))
    relative = np.abs(empirical - analytic) / np.max(np.abs(analytic))
    assert np.max(relative) < 0.12


def test_optimal_covariance_refuses_a_numerically_singular_fisher_matrix():
    # Good geometry, but the second sensor is 1e9 times worse, so the axis it
    # alone constrains carries no usable information.
    obs = VectorObservations(
        [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]], sigmas=[1e-6, 1e3]
    )
    with pytest.raises(DegenerateObservationsError, match="Fisher information"):
        optimal_covariance(obs)
