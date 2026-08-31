"""The four solvers: exactness, agreement, diagnostics, degeneracy behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import synthetic_problem, well_conditioned_reference
from wahbakit import (
    AttitudeSolution,
    DegenerateObservationsError,
    VectorObservations,
    angle_between_dcm,
    characteristic_coefficients,
    characteristic_polynomial,
    davenport_matrix,
    dcm_from_quat,
    is_rotation,
    olae,
    q_method,
    quest,
    quest_lambda_max,
    solve_wahba,
    triad,
    triad_frame,
    wahba_gain,
    wahba_loss,
)

METHOD_FUNCS = (q_method, quest, olae)


@pytest.mark.parametrize("solver", METHOD_FUNCS)
@pytest.mark.parametrize("n", [2, 3, 5, 12])
def test_noise_free_data_is_reproduced_exactly(solver, n):
    rng = np.random.default_rng(100 + n)
    obs, dcm = synthetic_problem(rng, n=n, sigma=0.0)
    solution = solver(obs)
    assert angle_between_dcm(solution.dcm, dcm) < 1e-11
    assert is_rotation(solution.dcm)


def test_triad_noise_free_data_is_reproduced_exactly():
    rng = np.random.default_rng(200)
    obs, dcm = synthetic_problem(rng, n=2, sigma=0.0)
    assert angle_between_dcm(triad(obs).dcm, dcm) < 1e-12


@pytest.mark.parametrize("solver", METHOD_FUNCS)
def test_returned_quaternion_matches_returned_dcm(solver):
    rng = np.random.default_rng(300)
    obs, _ = synthetic_problem(rng, n=4, sigma=1e-3)
    solution = solver(obs)
    assert np.allclose(dcm_from_quat(solution.quaternion), solution.dcm, atol=1e-13)
    assert solution.quaternion[0] >= 0.0


def test_all_methods_agree_on_well_conditioned_noisy_data():
    rng = np.random.default_rng(400)
    reference = well_conditioned_reference()
    obs, _ = synthetic_problem(rng, sigma=1e-4, reference=reference)
    optimal = q_method(obs)
    assert angle_between_dcm(quest(obs).dcm, optimal.dcm) < 1e-9
    # OLAE minimises a reweighted cost, so it differs at first order in sigma.
    assert angle_between_dcm(olae(obs).dcm, optimal.dcm) < 20 * 1e-4


def test_quest_lambda_max_matches_the_davenport_eigenvalue():
    rng = np.random.default_rng(500)
    for _ in range(20):
        obs, _ = synthetic_problem(rng, n=4, sigma=1e-2)
        eigenvalues = np.linalg.eigvalsh(davenport_matrix(obs.attitude_profile_matrix()))
        assert quest(obs).lambda_max == pytest.approx(float(eigenvalues[-1]), abs=1e-13)


def test_characteristic_polynomial_vanishes_at_every_davenport_eigenvalue():
    rng = np.random.default_rng(600)
    obs, _ = synthetic_problem(rng, n=5, sigma=1e-2)
    profile = obs.attitude_profile_matrix()
    coefficients = characteristic_coefficients(profile)
    for eigenvalue in np.linalg.eigvalsh(davenport_matrix(profile)):
        assert abs(characteristic_polynomial(float(eigenvalue), coefficients)) < 1e-13


def test_lambda_max_is_one_for_noise_free_data():
    rng = np.random.default_rng(700)
    obs, _ = synthetic_problem(rng, n=4, sigma=0.0)
    assert q_method(obs).lambda_max == pytest.approx(1.0, abs=1e-14)
    assert q_method(obs).loss == pytest.approx(0.0, abs=1e-14)


def test_optimal_methods_beat_triad_on_the_wahba_loss():
    rng = np.random.default_rng(800)
    obs, _ = synthetic_problem(rng, n=2, sigma=5e-2)
    assert q_method(obs).loss <= triad(obs).loss + 1e-15


def test_davenport_matrix_is_symmetric_with_the_right_trace():
    rng = np.random.default_rng(900)
    obs, _ = synthetic_problem(rng, n=4, sigma=1e-3)
    k = davenport_matrix(obs.attitude_profile_matrix())
    assert np.allclose(k, k.T)
    # trace K = trace(S) - 3 sigma + sigma = 2 sigma - 3 sigma + sigma = 0.
    assert float(np.trace(k)) == pytest.approx(0.0, abs=1e-14)


def test_davenport_matrix_rejects_bad_shape():
    with pytest.raises(ValueError, match="3, 3"):
        davenport_matrix(np.eye(2))


# -- 180 degree rotations and sequential rotation --------------------------
@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
@pytest.mark.parametrize("solver", [quest, olae])
def test_pi_rotation_is_recovered_with_sequential_rotation(solver, axis):
    unit_axis = np.array(axis, dtype=float)
    unit_axis /= np.linalg.norm(unit_axis)
    dcm = dcm_from_quat([0.0, *unit_axis])
    reference = well_conditioned_reference()
    obs = VectorObservations(reference @ dcm.T, reference)
    assert angle_between_dcm(solver(obs).dcm, dcm) < 1e-9


def test_quest_without_sequential_rotation_raises_at_exactly_pi():
    # Eq. Q3 is 0/0 there: gamma and X vanish together.
    dcm = dcm_from_quat([0.0, 1.0, 0.0, 0.0])
    reference = well_conditioned_reference()
    obs = VectorObservations(reference @ dcm.T, reference)
    with pytest.raises(RuntimeError, match="every candidate frame"):
        quest(obs, sequential_rotation=False)


def test_quest_without_sequential_rotation_degrades_approaching_pi():
    # Error grows like 1 / (pi - theta) without the pre-rotation and stays at
    # round-off with it.
    reference = well_conditioned_reference()
    errors = []
    for gap in (1e-2, 1e-6, 1e-10):
        angle = np.pi - gap
        dcm = dcm_from_quat([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        obs = VectorObservations(reference @ dcm.T, reference)
        errors.append(angle_between_dcm(quest(obs, sequential_rotation=False).dcm, dcm))
        assert angle_between_dcm(quest(obs).dcm, dcm) < 1e-12
    assert errors == sorted(errors)
    assert errors[-1] > 1e-8


def test_olae_without_sequential_rotation_raises_at_pi():
    dcm = dcm_from_quat([0.0, 0.0, 1.0, 0.0])
    reference = well_conditioned_reference()
    obs = VectorObservations(reference @ dcm.T, reference)
    with pytest.raises(RuntimeError, match="singular"):
        olae(obs, sequential_rotation=False)


# -- degeneracy ------------------------------------------------------------
def _near_parallel(separation_deg: float) -> VectorObservations:
    eta = np.radians(separation_deg)
    reference = np.array([[1.0, 0.0, 0.0], [np.cos(eta), np.sin(eta), 0.0]])
    dcm = dcm_from_quat([0.6, 0.4, -0.5, 0.3])
    return VectorObservations(reference @ dcm.T, reference, sigmas=[1e-3, 1e-3])


@pytest.mark.parametrize("solver", [triad, q_method, quest, olae])
def test_near_parallel_observations_raise(solver):
    obs = _near_parallel(0.05)
    with pytest.raises(DegenerateObservationsError, match="degenerate"):
        solver(obs)


@pytest.mark.parametrize("solver", [triad, q_method, quest, olae])
def test_degeneracy_check_can_be_disabled(solver):
    obs = _near_parallel(0.05)
    solution = solver(obs, check_degeneracy=False)
    assert isinstance(solution, AttitudeSolution)
    assert solution.observability is None


def test_degeneracy_error_message_names_the_angle():
    obs = _near_parallel(0.05)
    with pytest.raises(DegenerateObservationsError) as excinfo:
        q_method(obs)
    assert "0.0500" in str(excinfo.value)
    assert "lambda_min" in str(excinfo.value)


def test_exactly_parallel_observations_raise():
    obs = VectorObservations([[1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]])
    with pytest.raises(DegenerateObservationsError):
        q_method(obs)


def test_q_method_eigenvalue_gap_shrinks_as_geometry_degenerates():
    gaps = [
        q_method(_near_parallel(deg), check_degeneracy=False).diagnostics["eigenvalue_gap"]
        for deg in (90.0, 10.0, 1.0, 0.1)
    ]
    assert gaps == sorted(gaps, reverse=True)


def test_quest_lambda_max_raises_on_a_double_root():
    # sigma = kappa = ... chosen so psi'(lambda) vanishes: an all-zero profile.
    coefficients = characteristic_coefficients(np.zeros((3, 3)))
    with pytest.raises(RuntimeError, match="near-double root"):
        quest_lambda_max(coefficients, 0.0)


def test_quest_lambda_max_raises_when_iteration_budget_is_exhausted():
    rng = np.random.default_rng(1000)
    obs, _ = synthetic_problem(rng, n=4, sigma=1e-2)
    coefficients = characteristic_coefficients(obs.attitude_profile_matrix())
    with pytest.raises(RuntimeError, match="did not converge"):
        quest_lambda_max(coefficients, 1.0, tol=1e-300, max_iter=2)


# -- TRIAD specifics -------------------------------------------------------
def test_triad_reproduces_the_primary_observation_exactly():
    rng = np.random.default_rng(1100)
    obs, _ = synthetic_problem(rng, n=2, sigma=1e-2)
    for primary in (0, 1):
        dcm = triad(obs, primary=primary).dcm
        assert np.allclose(dcm @ obs.reference[primary], obs.body[primary], atol=1e-14)


def test_triad_choice_of_primary_changes_the_answer_under_noise():
    rng = np.random.default_rng(1200)
    obs, _ = synthetic_problem(rng, n=2, sigma=1e-2)
    assert angle_between_dcm(triad(obs, primary=0).dcm, triad(obs, primary=1).dcm) > 1e-6


def test_triad_rejects_more_than_two_observations():
    rng = np.random.default_rng(1300)
    obs, _ = synthetic_problem(rng, n=3, sigma=0.0)
    with pytest.raises(ValueError, match="exactly 2 observations"):
        triad(obs)


def test_triad_rejects_bad_primary_index():
    rng = np.random.default_rng(1400)
    obs, _ = synthetic_problem(rng, n=2, sigma=0.0)
    with pytest.raises(ValueError, match="primary must be 0 or 1"):
        triad(obs, primary=2)


def test_triad_frame_is_orthonormal_and_right_handed():
    frame = triad_frame([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-15)
    assert float(np.linalg.det(frame)) == pytest.approx(1.0, abs=1e-15)


def test_triad_frame_rejects_parallel_inputs():
    with pytest.raises(DegenerateObservationsError, match="parallel"):
        triad_frame([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])


# -- loss, gain, dispatcher ------------------------------------------------
def test_loss_and_gain_are_complementary():
    rng = np.random.default_rng(1500)
    obs, _ = synthetic_problem(rng, n=4, sigma=1e-2)
    solution = q_method(obs)
    assert wahba_loss(solution.dcm, obs) + wahba_gain(solution.dcm, obs) == pytest.approx(1.0)
    assert solution.loss == pytest.approx(wahba_loss(solution.dcm, obs))


def test_optimal_solution_minimises_the_loss_against_perturbations():
    rng = np.random.default_rng(1600)
    obs, _ = synthetic_problem(rng, n=4, sigma=1e-2)
    best = q_method(obs)
    for _ in range(50):
        perturbation = dcm_from_quat([1.0, *(1e-3 * rng.normal(size=3))])
        assert wahba_loss(perturbation @ best.dcm, obs) >= best.loss - 1e-18


def test_wahba_gain_rejects_bad_shape():
    rng = np.random.default_rng(1700)
    obs, _ = synthetic_problem(rng, n=3, sigma=0.0)
    with pytest.raises(ValueError, match="3, 3"):
        wahba_gain(np.eye(2), obs)


@pytest.mark.parametrize("method", ["q-method", "davenport", "quest", "olae"])
def test_solve_wahba_dispatches(method):
    rng = np.random.default_rng(1800)
    obs, dcm = synthetic_problem(rng, n=4, sigma=0.0)
    assert angle_between_dcm(solve_wahba(obs, method).dcm, dcm) < 1e-11


def test_solve_wahba_triad_and_kwargs():
    rng = np.random.default_rng(1900)
    obs, dcm = synthetic_problem(rng, n=2, sigma=0.0)
    assert angle_between_dcm(solve_wahba(obs, "triad", primary=1).dcm, dcm) < 1e-12


def test_solve_wahba_rejects_unknown_method():
    rng = np.random.default_rng(2000)
    obs, _ = synthetic_problem(rng, n=3, sigma=0.0)
    with pytest.raises(ValueError, match="unknown method"):
        solve_wahba(obs, "quaternion-magic")


def test_solve_wahba_with_covariance_returns_a_pair():
    rng = np.random.default_rng(2100)
    obs, _ = synthetic_problem(rng, n=4, sigma=1e-3)
    solution, covariance = solve_wahba(obs, "quest", with_covariance=True)
    assert covariance.shape == (3, 3)
    assert solution.method == "quest"


def test_solution_helpers():
    rng = np.random.default_rng(2200)
    obs, dcm = synthetic_problem(rng, n=4, sigma=1e-3)
    solution = quest(obs)
    assert solution.n_observations == 4
    assert solution.residual_angles_deg.shape == (4,)
    assert np.allclose(solution.rotate(obs.reference), obs.reference @ solution.dcm.T)
    assert solution.angle_to(dcm) == pytest.approx(angle_between_dcm(solution.dcm, dcm))
    assert "quest" in repr(solution)
    with pytest.raises(ValueError, match="N, 3"):
        solution.rotate(np.zeros((4, 2)))


def test_scipy_align_vectors_agreement():
    scipy_transform = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(2300)
    for _ in range(25):
        obs, _ = synthetic_problem(rng, n=5, sigma=1e-2)
        scipy_rotation, _ = scipy_transform.Rotation.align_vectors(
            obs.body, obs.reference, weights=obs.weights
        )
        assert angle_between_dcm(q_method(obs).dcm, scipy_rotation.as_matrix()) < 1e-10
