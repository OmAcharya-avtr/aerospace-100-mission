"""VectorObservations: validation, weights, profile matrix, observability."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import synthetic_problem
from wahbakit import DegenerateObservationsError, VectorObservations


def test_vectors_are_normalised_on_input():
    obs = VectorObservations([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0]], [[0.0, 0.0, 2.0], [7.0, 0.0, 0.0]])
    assert np.allclose(np.linalg.norm(obs.body, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(obs.reference, axis=1), 1.0)


def test_weights_default_to_equal_and_sum_to_one():
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]])
    assert np.allclose(obs.weights, [0.5, 0.5])


def test_weights_default_to_inverse_variance():
    # sigmas 1e-3 and 2e-3 -> weights 1/1e-6 : 1/4e-6 = 4 : 1 -> 0.8, 0.2.
    obs = VectorObservations(
        [[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]], sigmas=[1e-3, 2e-3]
    )
    assert np.allclose(obs.weights, [0.8, 0.2])


def test_explicit_weights_override_sigmas_and_are_normalised():
    obs = VectorObservations(
        [[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]], sigmas=[1e-3, 2e-3], weights=[3.0, 1.0]
    )
    assert np.allclose(obs.weights, [0.75, 0.25])
    assert np.allclose(obs.sigmas, [1e-3, 2e-3])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigmas": [1e-3]}, "sigmas must have 2 entries"),
        ({"sigmas": [1e-3, 0.0]}, "strictly positive"),
        ({"sigmas": [1e-3, -1.0]}, "strictly positive"),
        ({"weights": [1.0, 0.0]}, "strictly positive"),
        ({"weights": [1.0, 2.0, 3.0]}, "weights must have 2 entries"),
    ],
)
def test_invalid_sigmas_and_weights(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VectorObservations([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]], **kwargs)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="same shape"):
        VectorObservations([[1, 0, 0], [0, 1, 0]], [[0, 0, 1]])


def test_single_observation_rejected():
    with pytest.raises(ValueError, match="at least 2 observations"):
        VectorObservations([[1, 0, 0]], [[0, 0, 1]])


def test_require_sigmas_message_names_the_caller():
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]])
    assert not obs.has_sigmas
    with pytest.raises(ValueError, match="frobnication requires per-observation sigmas"):
        obs.require_sigmas("frobnication")


def test_attitude_profile_matrix_known_answer():
    # b1 = x, r1 = x; b2 = y, r2 = y; equal weights -> B = diag(0.5, 0.5, 0).
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]])
    assert np.allclose(obs.attitude_profile_matrix(), np.diag([0.5, 0.5, 0.0]))


def test_observability_orthogonal_pair_is_one_half():
    # Two orthogonal, equally weighted observations: lambda_min = (1 - 0) / 2.
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]])
    result = obs.observability()
    assert result.lambda_min == pytest.approx(0.5, abs=1e-14)
    assert result.min_separation_deg == pytest.approx(90.0, abs=1e-9)
    assert result.equivalent_separation_deg == pytest.approx(90.0, abs=1e-9)


@pytest.mark.parametrize("separation_deg", [30.0, 10.0, 1.0, 0.1])
def test_observability_matches_closed_form_for_a_pair(separation_deg):
    # lambda_min = (1 - |cos eta|) / 2 for two equally weighted observations.
    eta = np.radians(separation_deg)
    body = np.array([[1.0, 0.0, 0.0], [np.cos(eta), np.sin(eta), 0.0]])
    obs = VectorObservations(body, body)
    expected = (1.0 - abs(np.cos(eta))) / 2.0
    assert obs.observability().lambda_min == pytest.approx(expected, rel=1e-10)


def test_reference_frame_degeneracy_is_detected_separately():
    # Body vectors 90 deg apart, reference vectors nearly parallel.
    body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reference = np.array([[1.0, 0.0, 0.0], [1.0, 1e-5, 0.0]])
    obs = VectorObservations(body, reference)
    result = obs.observability()
    assert result.limiting_frame == "reference"
    with pytest.raises(DegenerateObservationsError, match="reference frame"):
        obs.require_observable()


def test_require_observable_rejects_non_positive_tol():
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="tol must be positive"):
        obs.require_observable(0.0)


def test_residual_angles_zero_for_the_true_attitude():
    rng = np.random.default_rng(11)
    obs, dcm = synthetic_problem(rng, n=5, sigma=0.0)
    assert np.max(obs.residual_angles(dcm)) < 1e-12


def test_residual_angles_rejects_bad_shape():
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="shape"):
        obs.residual_angles(np.eye(2))


def test_subset_renormalises_weights():
    obs = VectorObservations(
        np.eye(3), np.eye(3), sigmas=[1e-3, 2e-3, 4e-3]
    )
    pair = obs.subset([0, 1])
    assert pair.n == 2
    assert np.allclose(pair.weights, [0.8, 0.2])
    assert np.allclose(pair.sigmas, [1e-3, 2e-3])


def test_len_and_repr():
    obs = VectorObservations([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]])
    assert len(obs) == 2
    assert "VectorObservations(n=2" in repr(obs)


def test_observability_gate_ignores_the_weights():
    # Orthogonal directions with a 1000:1 sigma ratio are excellent geometry;
    # only the weighted diagnostic reflects the sensor imbalance.
    obs = VectorObservations(
        [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]], sigmas=[1e-6, 1e-3]
    )
    result = obs.observability()
    assert result.lambda_min == pytest.approx(0.5, abs=1e-14)
    assert result.weighted_lambda_min < 1e-5
    obs.require_observable()  # must not raise
